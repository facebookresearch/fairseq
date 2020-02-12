# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import Dictionary
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)


@register_model('dummy_model')
class DummyModel(FairseqLanguageModel):

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        parser.add_argument('--num-layers', type=int, default=24)
        parser.add_argument('--embed-dim', type=int, default=1024)

    @classmethod
    def build_model(cls, args, task):
        encoder = DummyEncoder(
            num_embed=len(task.target_dictionary),
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
        )
        return cls(args, encoder)

    def forward(self, src_tokens, **kwargs):
        return self.decoder(src_tokens)


class DummyEncoder(FairseqDecoder):

    def __init__(self, num_embed=50000, embed_dim=1024, num_layers=24):
        super().__init__(Dictionary())
        self.embed = nn.Embedding(
            num_embeddings=num_embed, embedding_dim=embed_dim, padding_idx=0
        )
        self.layers_a = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3*embed_dim),  # q, k, v input projection
                nn.Linear(3*embed_dim, embed_dim),  # skip self-attention
                nn.Linear(embed_dim, embed_dim),    # output projection
                nn.Dropout(),
            )
            for i in range(num_layers)
        ])
        self.layers_b = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 4*embed_dim),  # FFN
                nn.ReLU(),
                nn.Linear(4*embed_dim, embed_dim),  # FFN
                nn.Dropout(0.1),
            )
            for i in range(num_layers)
        ])
        self.out_proj = nn.Linear(embed_dim, num_embed)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer_a, layer_b in zip(self.layers_a, self.layers_b):
            x = x + layer_a(x)
            x = x + layer_b(x)
        x = self.out_proj(x)
        return (x,)

    def max_positions(self):
        return 1024

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


@register_model_architecture('dummy_model', 'dummy_model')
def base_architecture(args):
    pass
