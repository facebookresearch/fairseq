#!/usr/bin/env python3
import logging

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from typeguard import check_argument_types


from examples.speech_synthesis.preprocessing.tfgridnet.tasks.enh import EnhancementTask
from examples.speech_synthesis.preprocessing.tfgridnet.torch_utils.device_functions import to_device


def get_train_config(train_config, model_file=None):
    if train_config is None:
        assert model_file is not None, (
            "The argument 'model_file' must be provided "
            "if the argument 'train_config' is not specified."
        )
        train_config = Path(model_file).parent / "config.yaml"
    else:
        train_config = Path(train_config)
    return train_config


def recursive_dict_update(dict_org, dict_patch, verbose=False, log_prefix=""):
    """Update `dict_org` with `dict_patch` in-place recursively."""
    for key, value in dict_patch.items():
        if key not in dict_org:
            if verbose:
                logging.info(
                    "Overwriting config: [{}{}]: None -> {}".format(
                        log_prefix, key, value
                    )
                )
            dict_org[key] = value
        elif isinstance(value, dict):
            recursive_dict_update(
                dict_org[key], value, verbose=verbose, log_prefix=f"{key}."
            )
        else:
            if verbose and dict_org[key] != value:
                logging.info(
                    "Overwriting config: [{}{}]: {} -> {}".format(
                        log_prefix, key, dict_org[key], value
                    )
                )
            dict_org[key] = value


def build_model_from_args_and_file(task, args, model_file, device):
    model = task.build_model(args)
    if not isinstance(model, AbsESPnetModel):
        raise RuntimeError(
            f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
        )
    model.to(device)
    if model_file is not None:
        if device == "cuda":
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            device = f"cuda:{torch.cuda.current_device()}"
        model.load_state_dict(torch.load(model_file, map_location=device))
    return model


class SeparateSpeech:
    """SeparateSpeech class

    Examples:
        >>> import soundfile
        >>> separate_speech = SeparateSpeech("enh_config.yml", "enh.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> separate_speech(audio)
        [separated_audio1, separated_audio2, ...]

    """

    def __init__(
        self,
        train_config: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        inference_config: Union[Path, str] = None,
        segment_size: Optional[float] = None,
        hop_size: Optional[float] = None,
        normalize_segment_scale: bool = False,
        show_progressbar: bool = False,
        ref_channel: Optional[int] = None,
        normalize_output_wav: bool = False,
        device: str = "cpu",
        dtype: str = "float32",
        enh_s2t_task: bool = False,
    ):
        assert check_argument_types()

        task = EnhancementTask if not enh_s2t_task else EnhS2TTask

        # 1. Build Enh model

        if inference_config is None:
            enh_model, enh_train_args = task.build_model_from_file(
                train_config, model_file, device
            )
        else:
            # Overwrite model attributes
            train_config = get_train_config(train_config, model_file=model_file)
            with train_config.open("r", encoding="utf-8") as f:
                train_args = yaml.safe_load(f)

            with Path(inference_config).open("r", encoding="utf-8") as f:
                infer_args = yaml.safe_load(f)

            if enh_s2t_task:
                arg_list = ("enh_encoder", "enh_separator", "enh_decoder")
            else:
                arg_list = ("encoder", "separator", "decoder")
            supported_keys = list(chain(*[[k, k + "_conf"] for k in arg_list]))
            for k in infer_args.keys():
                if k not in supported_keys:
                    raise ValueError(
                        "Only the following top-level keys are supported: %s"
                        % ", ".join(supported_keys)
                    )

            recursive_dict_update(train_args, infer_args, verbose=True)
            enh_train_args = argparse.Namespace(**train_args)
            enh_model = build_model_from_args_and_file(
                task, enh_train_args, model_file, device
            )

        if enh_s2t_task:
            enh_model = enh_model.enh_model
        enh_model.to(dtype=getattr(torch, dtype)).eval()

        self.device = device
        self.dtype = dtype
        self.enh_train_args = enh_train_args
        self.enh_model = enh_model

        # only used when processing long speech, i.e.
        # segment_size is not None and hop_size is not None
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.normalize_segment_scale = normalize_segment_scale
        self.normalize_output_wav = normalize_output_wav
        self.show_progressbar = show_progressbar

        self.num_spk = enh_model.num_spk
        task = "enhancement" if self.num_spk == 1 else "separation"

        # reference channel for processing multi-channel speech
        if ref_channel is not None:
            logging.info(
                "Overwrite enh_model.separator.ref_channel with {}".format(ref_channel)
            )
            enh_model.separator.ref_channel = ref_channel
            if hasattr(enh_model.separator, "beamformer"):
                enh_model.separator.beamformer.ref_channel = ref_channel
            self.ref_channel = ref_channel
        else:
            self.ref_channel = enh_model.ref_channel

        self.segmenting = segment_size is not None and hop_size is not None
        if self.segmenting:
            logging.info("Perform segment-wise speech %s" % task)
            logging.info(
                "Segment length = {} sec, hop length = {} sec".format(
                    segment_size, hop_size
                )
            )
        else:
            logging.info("Perform direct speech %s on the input" % task)

    @torch.no_grad()
    def __call__(
        self, speech_mix: Union[torch.Tensor, np.ndarray], fs: int = 8000, **kwargs
    ) -> List[torch.Tensor]:
        """Inference

        Args:
            speech_mix: Input speech data (Batch, Nsamples [, Channels])
            fs: sample rate
        Returns:
            [separated_audio1, separated_audio2, ...]

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech_mix, np.ndarray):
            speech_mix = torch.as_tensor(speech_mix)

        assert speech_mix.dim() > 1, speech_mix.size()
        batch_size = speech_mix.size(0)
        speech_mix = speech_mix.to(getattr(torch, self.dtype))
        # lengths: (B,)
        lengths = speech_mix.new_full(
            [batch_size], dtype=torch.long, fill_value=speech_mix.size(1)
        )

        # a. To device
        speech_mix = to_device(speech_mix, device=self.device)
        lengths = to_device(lengths, device=self.device)

        ###################################
        # Normalize the signal variance
        if getattr(self.enh_model, "normalize_variance_per_ch", False):
            dim = 1
            mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
            speech_mix = speech_mix / mix_std_  # RMS normalization
        elif getattr(self.enh_model, "normalize_variance", False):
            if speech_mix.ndim > 2:
                dim = (1, 2)
            else:
                dim = 1
            mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
            speech_mix = speech_mix / mix_std_  # RMS normalization

        category = kwargs.get("utt2category", None)
        if (
            self.enh_model.categories
            and category is not None
            and category[0].item() not in self.enh_model.categories
        ):
            raise ValueError(f"Category '{category}' is not listed in self.categories")

        additional = {}
        if category is not None:
            cat = self.enh_model.categories[category[0].item()]
            print(f"category: {cat}", flush=True)
            if cat.endswith("_reverb"):
                additional["mode"] = "dereverb"
            else:
                additional["mode"] = "no_dereverb"

        if self.segmenting and lengths[0] > self.segment_size * fs:
            # Segment-wise speech enhancement/separation
            overlap_length = int(np.round(fs * (self.segment_size - self.hop_size)))
            num_segments = int(
                np.ceil((speech_mix.size(1) - overlap_length) / (self.hop_size * fs))
            )
            t = T = int(self.segment_size * fs)
            pad_shape = speech_mix[:, :T].shape
            enh_waves = []
            range_ = trange if self.show_progressbar else range
            for i in range_(num_segments):
                st = int(i * self.hop_size * fs)
                en = st + T
                if en >= lengths[0]:
                    # en - st < T (last segment)
                    en = lengths[0]
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = speech_mix[:, st:en]
                else:
                    t = T
                    speech_seg = speech_mix[:, st:en]  # B x T [x C]

                lengths_seg = speech_mix.new_full(
                    [batch_size], dtype=torch.long, fill_value=T
                )
                # b. Enhancement/Separation Forward
                feats, f_lens = self.enh_model.encoder(speech_seg, lengths_seg)
                feats, _, _ = self.enh_model.separator(feats, f_lens, additional)
                processed_wav = [
                    self.enh_model.decoder(f, lengths_seg)[0] for f in feats
                ]
                if speech_seg.dim() > 2:
                    # multi-channel speech
                    speech_seg_ = speech_seg[:, self.ref_channel]
                else:
                    speech_seg_ = speech_seg

                if self.normalize_segment_scale:
                    # normalize the scale to match the input mixture scale
                    mix_energy = torch.sqrt(
                        torch.mean(speech_seg_[:, :t].pow(2), dim=1, keepdim=True)
                    )
                    enh_energy = torch.sqrt(
                        torch.mean(
                            sum(processed_wav)[:, :t].pow(2), dim=1, keepdim=True
                        )
                    )
                    processed_wav = [
                        w * (mix_energy / enh_energy) for w in processed_wav
                    ]
                # List[torch.Tensor(num_spk, B, T)]
                enh_waves.append(torch.stack(processed_wav, dim=0))

            # c. Stitch the enhanced segments together
            waves = enh_waves[0]
            for i in range(1, num_segments):
                # permutation between separated streams in last and current segments
                perm = self.cal_permumation(
                    waves[:, :, -overlap_length:],
                    enh_waves[i][:, :, :overlap_length],
                    criterion="si_snr",
                )
                # repermute separated streams in current segment
                for batch in range(batch_size):
                    enh_waves[i][:, batch] = enh_waves[i][perm[batch], batch]

                if i == num_segments - 1:
                    enh_waves[i][:, :, t:] = 0
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

                # overlap-and-add (average over the overlapped part)
                waves[:, :, -overlap_length:] = (
                    waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
                ) / 2
                # concatenate the residual parts of the later segment
                waves = torch.cat([waves, enh_waves_res_i], dim=2)
            # ensure the stitched length is same as input
            assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
            waves = torch.unbind(waves, dim=0)
        else:
            # b. Enhancement/Separation Forward
            feats, f_lens = self.enh_model.encoder(speech_mix, lengths)
            feats, _, _ = self.enh_model.separator(feats, f_lens, additional)
            waves = [self.enh_model.decoder(f, lengths)[0] for f in feats]

        ###################################
        # De-normalize the signal variance
        if getattr(self.enh_model, "normalize_variance_per_ch", False):
            if mix_std_.ndim > 2:
                mix_std_ = mix_std_[:, :, self.ref_channel]
            waves = [w * mix_std_ for w in waves]
        elif getattr(self.enh_model, "normalize_variance", False):
            if mix_std_.ndim > 2:
                mix_std_ = mix_std_.squeeze(2)
            waves = [w * mix_std_ for w in waves]

        assert len(waves) == self.num_spk, len(waves) == self.num_spk
        assert len(waves[0]) == batch_size, (len(waves[0]), batch_size)
        if self.normalize_output_wav:
            waves = [
                (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy()
                for w in waves
            ]  # list[(batch, sample)]
        else:
            waves = [w.cpu().numpy() for w in waves]

        return waves

    @torch.no_grad()
    def cal_permumation(self, ref_wavs, enh_wavs, criterion="si_snr"):
        """Calculate the permutation between seaprated streams in two adjacent segments.

        Args:
            ref_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            enh_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            criterion (str): one of ("si_snr", "mse", "corr)
        Returns:
            perm (torch.Tensor): permutation for enh_wavs (Batch, num_spk)
        """

        criterion_class = {"si_snr": SISNRLoss, "mse": FrequencyDomainMSE}[criterion]

        pit_solver = PITSolver(criterion=criterion_class())

        _, _, others = pit_solver(ref_wavs, enh_wavs)
        perm = others["perm"]
        return perm

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build SeparateSpeech instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            SeparateSpeech: SeparateSpeech instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return SeparateSpeech(**kwargs)
