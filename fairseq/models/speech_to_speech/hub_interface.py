# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import Namespace
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import fairseq.data.audio.feature_transforms.utterance_cmvn as utt_cmvn
from fairseq import utils
from fairseq.data import encoders
from fairseq.data.audio.audio_utils import convert_waveform as convert_wav
from fairseq.data.audio.audio_utils import get_fbank
from fairseq.data.audio.audio_utils import get_waveform as get_wav
from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.data.audio.feature_transforms import (
    AUDIO_FEATURE_TRANSFORM_REGISTRY,
    CompositeAudioFeatureTransform,
)
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset

logger = logging.getLogger(__name__)

NUM_SPECIAL_TOKENS = 4


class S2SHubInterface(nn.Module):
    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.generator = self.task.build_generator([self.model], self.cfg.generation)
        eval_transforms = (
            S2TDataConfig(self.cfg.task.config_yaml)
            .config.get("transforms", {})
            .get("_eval")
        )
        eval_transforms = [
            AUDIO_FEATURE_TRANSFORM_REGISTRY[x].from_config_dict({})
            for x in eval_transforms
        ]
        self.feature_transforms = CompositeAudioFeatureTransform(eval_transforms)

    def postprocess_fbank(self, x):
        mean = x.mean(axis=0)
        square_sums = (x**2).sum(axis=0)
        x = np.subtract(x, mean)
        var = square_sums / x.shape[0] - mean**2
        std = np.sqrt(np.maximum(var, 1e-10))
        x = np.divide(x, std)
        return x

    def get_model_input(self, task, audio: str):
        feat = get_fbank(audio)  # C x T
        feat = self.feature_transforms(feat)  # ignore
        feat = torch.from_numpy(feat).float()
        src_tokens = self.postprocess_fbank(feat).unsqueeze(0)
        if torch.cuda.is_available():
            src_tokens = utils.move_to_cuda(src_tokens)
        src_lengths = torch.tensor(
            [src_tokens.size(1) for _ in range(src_tokens.size(0))], dtype=torch.long
        ).to(src_tokens)

        return {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": None,
                "tgt_speaker": None,
            },
            "target_lengths": None,
            "speaker": None,
        }

    @classmethod
    def get_prediction(
        cls, task, model, generator, sample, tgt_lang=None, synthesize_speech=False
    ) -> Union[str, Tuple[str, Tuple[torch.Tensor, int]]]:
        _tgt_lang = tgt_lang or task.data_cfg.hub.get("tgt_lang", None)
        pred_tokens = generator.generate([model], sample)
        pred = pred_tokens[0][0]["tokens"][:-1] - NUM_SPECIAL_TOKENS
        pred = " ".join([str(x) for x in pred.tolist()])

        if synthesize_speech:
            pfx = f"{_tgt_lang}_" if task.data_cfg.prepend_tgt_lang_tag else ""
            tts_model_id = task.data_cfg.hub.get(f"{pfx}tts_model_id", None)
            if tts_model_id is None:
                logger.warning("TTS model configuration not found")
            else:
                temp = tts_model_id.split(":")
                if len(temp) == 2:
                    _repo, _id = temp
                elif len(temp) == 3:
                    _repo, _id = ":".join(temp[:2]), temp[2]
                else:
                    raise Exception("Invalid TTS model path")
                tts_model = torch.hub.load(_repo, _id, verbose=False)
                pred = (pred, tts_model.predict(pred))
        return pred

    def predict(
        self,
        audio: str,
        tgt_lang: Optional[str] = None,
        synthesize_speech: bool = False,
    ) -> Union[str, Tuple[str, Tuple[torch.Tensor, int]]]:
        sample = self.get_model_input(self.task, audio)
        return self.get_prediction(
            self.task,
            self.model,
            self.generator,
            sample,
            tgt_lang=tgt_lang,
            synthesize_speech=synthesize_speech,
        )
