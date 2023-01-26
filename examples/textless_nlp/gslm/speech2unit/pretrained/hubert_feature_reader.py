# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import fairseq
import soundfile as sf
import torch.nn.functional as F


class HubertFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer, max_chunk=1600000, use_cuda=True):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0].eval()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

    def read_audio(self, path, ref_len=None, channel_id=None):
        wav, sr = sf.read(path)
        if channel_id is not None:
            assert wav.ndim == 2, \
                f"Expected stereo input when channel_id is given ({path})"
            assert channel_id in [1, 2], \
                "channel_id is expected to be in [1, 2]"
            wav = wav[:, channel_id-1]
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, ref_len=None, channel_id=None):
        x = self.read_audio(file_path, ref_len, channel_id)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            if self.use_cuda:
                x = x.cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)
