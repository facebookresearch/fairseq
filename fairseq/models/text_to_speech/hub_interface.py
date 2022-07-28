# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TTSHubInterface(nn.Module):
    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        self.model.eval()

        self.update_cfg_with_data_cfg(self.cfg, self.task.data_cfg)
        self.generator = self.task.build_generator([self.model], self.cfg)

    @classmethod
    def phonemize(
        cls,
        text: str,
        lang: Optional[str],
        phonemizer: Optional[str] = None,
        preserve_punct: bool = False,
        to_simplified_zh: bool = False,
    ):
        if to_simplified_zh:
            import hanziconv

            text = hanziconv.HanziConv.toSimplified(text)

        if phonemizer == "g2p":
            import g2p_en

            g2p = g2p_en.G2p()
            if preserve_punct:
                return " ".join("|" if p == " " else p for p in g2p(text))
            else:
                res = [{",": "sp", ";": "sp"}.get(p, p) for p in g2p(text)]
                return " ".join(p for p in res if p.isalnum())
        if phonemizer == "g2pc":
            import g2pc

            g2p = g2pc.G2pC()
            return " ".join([w[3] for w in g2p(text)])
        elif phonemizer == "ipa":
            assert lang is not None
            import phonemizer
            from phonemizer.separator import Separator

            lang_map = {"en": "en-us", "fr": "fr-fr"}
            return phonemizer.phonemize(
                text,
                backend="espeak",
                language=lang_map.get(lang, lang),
                separator=Separator(word="| ", phone=" "),
            )
        else:
            return text

    @classmethod
    def tokenize(cls, text: str, tkn_cfg: Dict[str, str]):
        sentencepiece_model = tkn_cfg.get("sentencepiece_model", None)
        if sentencepiece_model is not None:
            assert Path(sentencepiece_model).exists()
            import sentencepiece as sp

            spm = sp.SentencePieceProcessor()
            spm.Load(sentencepiece_model)
            return " ".join(spm.Encode(text, out_type=str))
        else:
            return text

    @classmethod
    def update_cfg_with_data_cfg(cls, cfg, data_cfg):
        cfg["task"].vocoder = data_cfg.vocoder.get("type", "griffin_lim")

    @classmethod
    def get_model_input(
        cls, task, text: str, speaker: Optional[int] = None, verbose: bool = False
    ):
        phonemized = cls.phonemize(
            text,
            task.data_cfg.hub.get("lang", None),
            task.data_cfg.hub.get("phonemizer", None),
            task.data_cfg.hub.get("preserve_punct", False),
            task.data_cfg.hub.get("to_simplified_zh", False),
        )
        tkn_cfg = task.data_cfg.bpe_tokenizer
        tokenized = cls.tokenize(phonemized, tkn_cfg)
        if verbose:
            logger.info(f"text: {text}")
            logger.info(f"phonemized: {phonemized}")
            logger.info(f"tokenized: {tokenized}")

        spk = task.data_cfg.hub.get("speaker", speaker)
        n_speakers = len(task.speaker_to_id or {})
        if spk is None and n_speakers > 0:
            spk = random.randint(0, n_speakers - 1)
        if spk is not None:
            spk = max(0, min(spk, n_speakers - 1))
        if verbose:
            logger.info(f"speaker: {spk}")
        spk = None if spk is None else torch.Tensor([[spk]]).long()

        src_tokens = task.src_dict.encode_line(tokenized, add_if_not_exist=False).view(
            1, -1
        )
        src_lengths = torch.Tensor([len(tokenized.split())]).long()
        return {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": None,
            },
            "target_lengths": None,
            "speaker": spk,
        }

    @classmethod
    def get_prediction(cls, task, model, generator, sample) -> Tuple[torch.Tensor, int]:
        prediction = generator.generate(model, sample)
        return prediction[0]["waveform"], task.sr

    def predict(
        self, text: str, speaker: Optional[int] = None, verbose: bool = False
    ) -> Tuple[torch.Tensor, int]:
        sample = self.get_model_input(self.task, text, speaker, verbose=verbose)
        return self.get_prediction(self.task, self.model, self.generator, sample)


class VocoderHubInterface(nn.Module):
    """Vocoder interface to run vocoder models through hub. Currently we only support unit vocoder"""

    def __init__(self, cfg, model):
        super().__init__()
        self.vocoder = model
        self.vocoder.eval()
        self.sr = 16000
        self.multispkr = self.vocoder.model.multispkr
        if self.multispkr:
            logger.info("multi-speaker vocoder")
            self.num_speakers = cfg.get(
                "num_speakers",
                200,
            )  # following the default in codehifigan to set to 200

    def get_model_input(
        self,
        text: str,
        speaker: Optional[int] = -1,
    ):
        units = list(map(int, text.strip().split()))
        x = {
            "code": torch.LongTensor(units).view(1, -1),
        }
        if not speaker:
            speaker = -1
        if self.multispkr:
            assert (
                speaker < self.num_speakers
            ), f"invalid --speaker-id ({speaker}) with total #speakers = {self.num_speakers}"
            spk = random.randint(0, self.num_speakers - 1) if speaker == -1 else speaker
            x["spkr"] = torch.LongTensor([spk]).view(1, 1)
        return x

    def get_prediction(self, sample, dur_prediction: Optional[bool] = True):
        wav = self.vocoder(sample, dur_prediction)
        return wav, self.sr

    def predict(
        self,
        text: str,
        speaker: Optional[int] = None,
        dur_prediction: Optional[bool] = True,
    ):
        sample = self.get_model_input(text, speaker)
        return self.get_prediction(sample, dur_prediction)
