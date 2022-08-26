# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import logging
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import encoders
from fairseq.data.audio.audio_utils import (
    get_waveform as get_wav,
    convert_waveform as convert_wav,
    get_fbank,
)
import fairseq.data.audio.feature_transforms.utterance_cmvn as utt_cmvn
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset

logger = logging.getLogger(__name__)


class S2THubInterface(nn.Module):
    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        self.model.eval()
        self.generator = self.task.build_generator([self.model], self.cfg.generation)

    @classmethod
    def get_model_input(cls, task, audio: Union[str, torch.Tensor]):
        input_type = task.data_cfg.hub.get("input_type", "fbank80")
        if input_type == "fbank80_w_utt_cmvn":
            if isinstance(audio, str):
                feat = utt_cmvn.UtteranceCMVN()(get_fbank(audio))
                feat = feat.unsqueeze(0)  # T x D -> 1 x T x D
            else:
                import torchaudio.compliance.kaldi as kaldi

                feat = kaldi.fbank(audio, num_mel_bins=80).numpy()  # 1 x T x D
        elif input_type in {"waveform", "standardized_waveform"}:
            if isinstance(audio, str):
                feat, sr = get_wav(audio)  # C x T
                feat, _ = convert_wav(
                    feat, sr, to_sample_rate=16_000, to_mono=True
                )  # C x T -> 1 x T
            else:
                feat = audio.numpy()
        else:
            raise ValueError(f"Unknown value: input_type = {input_type}")

        src_lengths = torch.Tensor([feat.shape[1]]).long()
        src_tokens = torch.from_numpy(feat)  # 1 x T (x D)
        if input_type == "standardized_waveform":
            with torch.no_grad():
                src_tokens = F.layer_norm(src_tokens, src_tokens.shape)

        return {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": None,
            },
            "target_lengths": None,
            "speaker": None,
        }

    @classmethod
    def detokenize(cls, task, tokens):
        text = task.tgt_dict.string(tokens)
        tkn_cfg = task.data_cfg.bpe_tokenizer
        tokenizer = encoders.build_bpe(Namespace(**tkn_cfg))
        return text if tokenizer is None else tokenizer.decode(text)

    @classmethod
    def get_prefix_token(cls, task, lang):
        prefix_size = int(task.data_cfg.prepend_tgt_lang_tag)
        prefix_tokens = None
        if prefix_size > 0:
            assert lang is not None
            lang_tag = SpeechToTextDataset.get_lang_tag_idx(lang, task.tgt_dict)
            prefix_tokens = torch.Tensor([lang_tag]).long().unsqueeze(0)
        return prefix_tokens

    @classmethod
    def get_prediction(
        cls, task, model, generator, sample, tgt_lang=None, synthesize_speech=False
    ) -> Union[str, Tuple[str, Tuple[torch.Tensor, int]]]:
        _tgt_lang = tgt_lang or task.data_cfg.hub.get("tgt_lang", None)
        prefix = cls.get_prefix_token(task, _tgt_lang)
        pred_tokens = generator.generate([model], sample, prefix_tokens=prefix)
        pred = cls.detokenize(task, pred_tokens[0][0]["tokens"])
        eos_token = task.data_cfg.config.get("eos_token", None)
        if eos_token:
            pred = ' '.join(pred.split(' ')[:-1])

        if synthesize_speech:
            pfx = f"{_tgt_lang}_" if task.data_cfg.prepend_tgt_lang_tag else ""
            tts_model_id = task.data_cfg.hub.get(f"{pfx}tts_model_id", None)
            speaker = task.data_cfg.hub.get(f"{pfx}speaker", None)
            if tts_model_id is None:
                logger.warning("TTS model configuration not found")
            else:
                _repo, _id = tts_model_id.split(":")
                tts_model = torch.hub.load(_repo, _id, verbose=False)
                pred = (pred, tts_model.predict(pred, speaker=speaker))
        return pred

    def predict(
        self,
        audio: Union[str, torch.Tensor],
        tgt_lang: Optional[str] = None,
        synthesize_speech: bool = False,
    ) -> Union[str, Tuple[str, Tuple[torch.Tensor, int]]]:
        # `audio` is either a file path or a 1xT Tensor
        # return either text or (text, synthetic speech)
        sample = self.get_model_input(self.task, audio)
        return self.get_prediction(
            self.task,
            self.model,
            self.generator,
            sample,
            tgt_lang=tgt_lang,
            synthesize_speech=synthesize_speech,
        )
