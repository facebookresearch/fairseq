# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2_asr import (
    Wav2Vec2CtcConfig,
    Wav2VecCtc,
    Wav2VecEncoder,
)
from fairseq.tasks import FairseqTask


@register_model("wav2vec2_laser", dataclass=Wav2Vec2CtcConfig)
class Wav2VecLaser(Wav2VecCtc):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)
        self.num_updates = 0
        self.freeze_finetune_updates = cfg.freeze_finetune_updates

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, 1024)
        return cls(cfg, w2v_encoder)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        x_out = output["encoder_out"] * 0.01
        out_pad_mask = output["padding_mask"]
        # Set padded outputs to -inf so they are not selected by max-pooling
        if out_pad_mask is not None and out_pad_mask.any():
            x_out = (
                x_out.float()
                .masked_fill_(out_pad_mask.T.unsqueeze(-1), float("-inf"))
                .type_as(x_out)
            )
        return x_out.max(dim=0)[0]
