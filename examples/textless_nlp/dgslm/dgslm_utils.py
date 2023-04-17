# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import json

from fairseq import utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

# from examples.hubert.simple_kmeans.dump_hubert_feature import HubertFeatureReader
from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader
from examples.hubert.simple_kmeans.dump_km_label import ApplyKmeans


# Hubert tokenizer
class HubertTokenizer:
    def __init__(
                self,
                hubert_path,
                hubert_layer,
                km_path,
                use_cuda=True,
            ):
        self.feature_extractor = HubertFeatureReader(hubert_path, hubert_layer, use_cuda=use_cuda)
        self.quantizer = ApplyKmeans(km_path)
        if not use_cuda:
            self.quantizer.C = self.quantizer.C.cpu()
            self.quantizer.Cnorm = self.quantizer.Cnorm.cpu()

    def wav2code(self, path, channel_id=1):
        feat = self.feature_extractor.get_feats(path, channel_id=channel_id)
        code = self.quantizer(feat)
        return ' '.join(map(str, code))

    def wav2codes(self, path):
        codes = [
            self.wav2code(path, channel_id=1),
            self.wav2code(path, channel_id=2)
        ]
        return codes


# Vocoder
class HifiganVocoder:
    def __init__(
                self,
                vocoder_path,
                vocoder_cfg_path,
                use_cuda=True,
            ):
        with open(vocoder_cfg_path) as f:
            cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(vocoder_path, cfg).eval()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.vocoder.cuda()

    def code2wav(self, code, speaker_id=0, pred_dur=False):
        if isinstance(code, str):
            code = list(map(int, code.split()))
        inp = {"code": torch.LongTensor(code).view(1, -1)}
        if self.vocoder.model.multispkr:
            inp["spkr"] = torch.LongTensor([speaker_id]).view(1, 1)
        if self.use_cuda:
            inp = utils.move_to_cuda(inp)
        return self.vocoder(inp, pred_dur).detach().cpu().numpy()

    def codes2wav(self, codes, speaker_ids=[0, 4], pred_dur=False):
        if isinstance(codes, dict):
            codes = list(codes.values())
        assert len(codes) == 2
        wav1 = self.code2wav(codes[0], speaker_ids[0], pred_dur)
        wav2 = self.code2wav(codes[1], speaker_ids[1], pred_dur)
        wav = np.stack([wav1, wav2])
        return wav
