import json
import re
import urllib.request
from pathlib import Path

import fairseq
import torch
from fairseq.data.data_utils import lengths_to_padding_mask
from tqdm import tqdm

try:
    import torchaudio
    from torchaudio.models.decoder import ctc_decoder
except ImportError:
    raise ImportError("Upgrade torchaudio to 0.12 to enable CTC decoding")


class DownloadProgressBar(tqdm):
    """A class to represent a download progress bar"""

    def update_to(self, b=1, bsize=1, tsize=None) -> None:
        """
        Update the download progress
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def retrieve_asr_config(lang_key: str, asr_version: str, json_path: str) -> dict:
    """
    Retrieve the asr model configs

    Args:
        lang_key: the lanuage type as the key name
        json_path: the path of the config json file

    Returns:
        Dict of all the configs in the json file
    """

    with open(json_path, "r") as f:
        asr_model_cfgs = json.load(f)
    return asr_model_cfgs[lang_key][asr_version]


class ASRGenerator(object):
    """A class to represent a ASR generator"""

    def __init__(
        self,
        model_cfg: dict,
        cache_dirpath: str = (Path.home() / ".cache" / "ust_asr").as_posix(),
    ) -> None:
        """
        Construct all the necessary attributes of the ASRGenerator class

        Args:
            model_cfg: the dict of the asr model config
            cache_dirpath: the default cache path is "Path.home()/.cache/ust_asr"
        """

        self.cache_dirpath = Path(cache_dirpath) / model_cfg["lang"]
        self.model_cfg = model_cfg

        self.use_cuda = torch.cuda.is_available()

        torchaudio.set_audio_backend("sox_io")

        if self.model_cfg["model_type"] == "hf":
            self.prepare_hf_model(self.model_cfg)
        elif self.model_cfg["model_type"] == "fairseq":
            self.prepare_fairseq_model(self.model_cfg)
        else:
            raise NotImplementedError(
                f"Model type {self.model_cfg['model_type']} is not supported"
            )

        if self.model_cfg["post_process"] == "collapse":
            self.post_process_fn = lambda hypo: "".join(hypo).replace(
                self.sil_token, " "
            )
        elif self.model_cfg["post_process"] == "none":
            self.post_process_fn = lambda hypo: " ".join(hypo).replace(
                self.sil_token, " "
            )
        else:
            raise NotImplementedError

        if self.use_cuda:
            self.model.cuda()
        self.model.eval()

        self.decoder = ctc_decoder(
            lexicon=None,
            tokens=self.tokens,
            lm=None,
            nbest=1,
            beam_size=1,
            beam_size_token=None,
            lm_weight=0.0,
            word_score=0.0,
            unk_score=float("-inf"),
            sil_token=self.sil_token,
            sil_score=0.0,
            log_add=False,
            blank_token=self.blank_token,
        )

    def prepare_hf_model(self, model_cfg: dict) -> None:
        """
        Prepare the huggingface asr model

        Args:
            model_cfg: dict with the relevant ASR config
        """

        def infer_silence_token(vocab: list):
            """
            Different HF checkpoints have different notion of silence token
            such as | or " " (space)
            Important: when adding new HF asr model in, check what silence token it uses
            """
            if "|" in vocab:
                return "|"
            elif " " in vocab:
                return " "
            else:
                raise RuntimeError("Silence token is not found in the vocabulary")

        try:
            from transformers import (AutoFeatureExtractor, AutoTokenizer,
                                      Wav2Vec2ForCTC, Wav2Vec2Processor)
        except ImportError:
            raise ImportError("Install transformers to load HF wav2vec model")

        model_path = model_cfg["model_path"]
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.preprocessor = AutoFeatureExtractor.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)

        # extra unk tokens are there to make some models work e.g. Finnish ASR has some vocab issue
        vocab_list = [
            self.tokenizer.decoder.get(i, f"{self.tokenizer.unk_token}1")
            for i in range(self.tokenizer.vocab_size)
        ]

        self.sampling_rate = self.preprocessor.sampling_rate
        self.normalize_input = self.preprocessor.do_normalize
        self.tokens = vocab_list
        self.sil_token = infer_silence_token(vocab_list)
        self.blank_token = self.tokenizer.pad_token

    def prepare_fairseq_model(self, model_cfg: dict) -> None:
        """
        Prepare the fairseq asr model

        Args:
            model_cfg: the specific model config dict must have: (1) ckpt_path, (2) dict_path
        """

        def download_file(url: str, cache_dir: Path):
            download_path = cache_dir / url.split("/")[-1]
            if not (cache_dir / url.split("/")[-1]).exists():
                with DownloadProgressBar(
                    unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
                ) as t:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(
                        url, filename=download_path.as_posix(), reporthook=t.update_to
                    )
            else:
                print(f"'{url}' exists in {cache_dir}")

            return download_path.as_posix()

        try:
            ckpt_path = model_cfg["ckpt_path"]
            dict_path = model_cfg["dict_path"]
        except KeyError:
            raise KeyError(
                "Fairseq model cfg must provide (1) ckpt_path, (2) dict_path"
            )

        if re.search("^https", ckpt_path):
            ckpt_path = download_file(ckpt_path, self.cache_dirpath)
        if re.search("^https", dict_path):
            dict_path = download_file(dict_path, self.cache_dirpath)

        model, saved_cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path],
            arg_overrides={
                "task": "audio_finetuning",
                "data": self.cache_dirpath.as_posix(),
            },  # data must have dict in it
        )

        dict_lines = open(dict_path, "r").readlines()
        tokens = [l.split()[0] for l in dict_lines]
        # adding default fairseq special tokens
        tokens = ["<s>", "<pad>", "</s>", "<unk>"] + tokens

        self.model = model[0]
        self.tokens = tokens

        if "|" in tokens:
            self.sil_token = "|"
        else:
            self.sil_token = tokens[
                2
            ]  # use eos as silence token if | not presented e.g., Hok ASR model
        print(f"Inferring silence token from the dict: {self.sil_token}")
        self.blank_token = self.tokens[0]

        self.sampling_rate = saved_cfg.task.sample_rate
        self.normalize_input = saved_cfg.task.normalize

    @torch.inference_mode()
    def load_audiofile(self, audio_path: str) -> torch.Tensor:
        """
        Load the audio files and apply resampling and normalizaion

        Args:
            audio_path: the audio file path

        Returns:
            audio_waveform: the audio waveform as a torch.Tensor object
        """

        audio_waveform, sampling_rate = torchaudio.load(audio_path)
        if audio_waveform.dim == 2:
            audio_waveform = audio_waveform.mean(-1)
        if self.sampling_rate != sampling_rate:
            audio_waveform = torchaudio.functional.resample(
                audio_waveform, sampling_rate, self.sampling_rate
            )
        if self.normalize_input:
            # following fairseq raw audio dataset
            audio_waveform = torch.nn.functional.layer_norm(
                audio_waveform, audio_waveform.shape
            )

        return audio_waveform

    @torch.inference_mode()
    def compute_emissions(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Compute the emissions for either fairseq or huggingface asr model

        Args:
            audio_path: the input audio waveform

        Returns:
            emissions: the logits of the encoded prediction.
        """

        if self.use_cuda:
            audio_input = audio_input.to("cuda")
        if isinstance(self.model, fairseq.models.wav2vec.wav2vec2_asr.Wav2VecCtc):
            padding_mask = lengths_to_padding_mask(torch.tensor([audio_input.numel()]))
            emissions = self.model.w2v_encoder(audio_input, padding_mask)[
                "encoder_out"
            ].transpose(0, 1)
        else:
            emissions = self.model(audio_input).logits

        return emissions

    def decode_emissions(self, emissions: torch.Tensor) -> str:
        """
        Decode the emissions and apply post process functions

        Args:
            emissions: the input Tensor object

        Returns:
            hypo: the str as the decoded transcriptions
        """

        emissions = emissions.cpu()
        results = self.decoder(emissions)

        # assuming the lexicon-free decoder and working with tokens
        hypo = self.decoder.idxs_to_tokens(results[0][0].tokens)
        hypo = self.post_process_fn(hypo)

        return hypo

    def transcribe_audiofile(self, audio_path: str, lower=True) -> str:
        """
        Transcribe the audio into string

        Args:
            audio_path: the input audio waveform
            lower: the case of the transcriptions with lowercase as the default

        Returns:
            hypo: the transcription result
        """

        asr_input = self.load_audiofile(audio_path)
        emissions = self.compute_emissions(asr_input)
        hypo = self.decode_emissions(emissions)

        return hypo.strip().lower() if lower else hypo.strip()
