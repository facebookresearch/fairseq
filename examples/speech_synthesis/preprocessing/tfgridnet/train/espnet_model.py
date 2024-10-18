"""Enhancement model module."""
from typing import Dict, List, Optional, OrderedDict, Tuple
from abc import ABC, abstractmethod

import numpy as np
import torch
from typeguard import check_argument_types

from examples.speech_synthesis.preprocessing.tfgridnet.mask import AbsMask
from examples.speech_synthesis.preprocessing.tfgridnet.enh.decoder import AbsDecoder
from examples.speech_synthesis.preprocessing.tfgridnet.enh.encoder import AbsEncoder
from examples.speech_synthesis.preprocessing.tfgridnet.enh.wrappers import AbsLossWrapper
from examples.speech_synthesis.preprocessing.tfgridnet.enh.separator import AbsSeparator

#from packaging.version import parse as V
#is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

class AbsESPnetModel(torch.nn.Module, ABC):
   
    @abstractmethod
    def forward(
        self, **batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class ESPnetEnhancementModel(AbsESPnetModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        encoder: AbsEncoder,
        separator: AbsSeparator,
        decoder: AbsDecoder,
        mask_module: Optional[AbsMask],
        loss_wrappers: List[AbsLossWrapper],
        stft_consistency: bool = False,
        loss_type: str = "mask_mse",
        mask_type: Optional[str] = None,
        flexible_numspk: bool = False,
        extract_feats_in_collect_stats: bool = False,
        normalize_variance: bool = False,
        normalize_variance_per_ch: bool = False,
        categories: list = [],
        category_weights: list = [],
    ):
        
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder
        self.mask_module = mask_module
        self.num_spk = separator.num_spk
        # If True, self.num_spk is regarded as the MAXIMUM possible number of speakers
        self.flexible_numspk = flexible_numspk
        self.num_noise_type = getattr(self.separator, "num_noise_type", 1)

        self.loss_wrappers = loss_wrappers
        names = [w.criterion.name for w in self.loss_wrappers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicated loss names are not allowed: {}".format(names))

        # kept for compatibility
        self.mask_type = mask_type.upper() if mask_type else None
        self.loss_type = loss_type
        self.stft_consistency = stft_consistency

        # for multi-channel signal
        self.ref_channel = getattr(self.separator, "ref_channel", None)
        if self.ref_channel is None:
            self.ref_channel = 0

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.normalize_variance = normalize_variance
        self.normalize_variance_per_ch = normalize_variance_per_ch
        if normalize_variance and normalize_variance_per_ch:
            raise ValueError(
                "normalize_variance and normalize_variance_per_ch cannot be True "
                "at the same time."
            )

        # list all possible categories of the batch (order matters!)
        # (used to convert category index to the corresponding name for logging)
        self.categories = {}
        if categories:
            count = 0
            for c in categories:
                if c not in self.categories:
                    self.categories[count] = c
                    count += 1
        # used to set loss weights for batches of different categories
        if category_weights:
            assert len(category_weights) == len(self.categories)
            self.category_weights = tuple(category_weights)
        else:
            self.category_weights = tuple(1.0 for _ in self.categories)

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            kwargs: "utt_id" is among the input.
        """
        # reference speech signal of each speaker
        assert "speech_ref1" in kwargs, "At least 1 reference signal input is required."
        speech_ref = [
            kwargs.get(
                f"speech_ref{spk + 1}",
                torch.zeros_like(kwargs["speech_ref1"]),
            )
            for spk in range(self.num_spk)
            if f"speech_ref{spk + 1}" in kwargs
        ]
        num_spk = len(speech_ref) if self.flexible_numspk else self.num_spk
        assert len(speech_ref) == num_spk, (len(speech_ref), num_spk)
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)

        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using beamforming-based
            # frontend models)
            noise_ref = [
                kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
            # (Batch, num_noise_type, samples) or
            # (Batch, num_noise_type, samples, channels)
            noise_ref = torch.stack(noise_ref, dim=1)
        else:
            noise_ref = None

        # dereverberated (noisy) signal
        # (optional, only used for frontend models with WPE)
        if "dereverb_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            dereverb_speech_ref = [
                kwargs["dereverb_ref{}".format(n + 1)]
                for n in range(num_spk)
                if "dereverb_ref{}".format(n + 1) in kwargs
            ]
            assert len(dereverb_speech_ref) in (1, num_spk), len(dereverb_speech_ref)
            # (Batch, N, samples) or (Batch, N, samples, channels)
            dereverb_speech_ref = torch.stack(dereverb_speech_ref, dim=1)
        else:
            dereverb_speech_ref = None

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int().fill_(speech_mix.shape[1])
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )

        # for data-parallel
        speech_ref = speech_ref[..., : speech_lengths.max()].unbind(dim=1)
        if noise_ref is not None:
            noise_ref = noise_ref[..., : speech_lengths.max()].unbind(dim=1)
        if dereverb_speech_ref is not None:
            dereverb_speech_ref = dereverb_speech_ref[..., : speech_lengths.max()]
            dereverb_speech_ref = dereverb_speech_ref.unbind(dim=1)

        # sampling frequency information about the batch
        fs = None
        if "utt2fs" in kwargs:
            # All samples must have the same sampling rate
            fs = kwargs["utt2fs"][0].item()
            assert all([fs == kwargs["utt2fs"][0].item() for fs in kwargs["utt2fs"]])

            # Adaptively adjust the STFT/iSTFT window/hop sizes for USESSeparator
            if not isinstance(self.separator, USESSeparator):
                fs = None

        # category information (integer) about the batch
        category = kwargs.get("utt2category", None)
        if (
            self.categories
            and category is not None
            and category[0].item() not in self.categories
        ):
            raise ValueError(f"Category '{category}' is not listed in self.categories")

        additional = {}
        # Additional data is required in Deep Attractor Network
        if isinstance(self.separator, DANSeparator):
            additional["feature_ref"] = [
                self.encoder(r, speech_lengths, fs=fs)[0] for r in speech_ref
            ]
        if self.flexible_numspk:
            additional["num_spk"] = num_spk
        # Additional information is required in USES for multi-condition training
        if category is not None and isinstance(self.separator, USESSeparator):
            cat = self.categories[category[0].item()]
            if cat.endswith("_both"):
                additional["mode"] = "both"
            elif cat.endswith("_reverb"):
                additional["mode"] = "dereverb"
            else:
                additional["mode"] = "no_dereverb"

        speech_mix = speech_mix[:, : speech_lengths.max()]

        ###################################
        # Normalize the signal variance
        if self.normalize_variance_per_ch:
            dim = 1
            mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
            speech_mix = speech_mix / mix_std_  # RMS normalization
        elif self.normalize_variance:
            if speech_mix.ndim > 2:
                dim = (1, 2)
            else:
                dim = 1
            mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
            speech_mix = speech_mix / mix_std_  # RMS normalization

        # model forward
        speech_pre, feature_mix, feature_pre, others = self.forward_enhance(
            speech_mix, speech_lengths, additional, fs=fs
        )

        ###################################
        # De-normalize the signal variance
        if self.normalize_variance_per_ch and speech_pre is not None:
            if mix_std_.ndim > 2:
                mix_std_ = mix_std_[:, :, self.ref_channel]
            speech_pre = [sp * mix_std_ for sp in speech_pre]
        elif self.normalize_variance and speech_pre is not None:
            if mix_std_.ndim > 2:
                mix_std_ = mix_std_.squeeze(2)
            speech_pre = [sp * mix_std_ for sp in speech_pre]

        # loss computation
        loss, stats, weight, perm = self.forward_loss(
            speech_pre,
            speech_lengths,
            feature_mix,
            feature_pre,
            others,
            speech_ref,
            noise_ref,
            dereverb_speech_ref,
            category,
            num_spk=num_spk,
            fs=fs,
        )
        return loss, stats, weight

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
