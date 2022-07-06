# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn as nn

from fairseq.models.transformer import TransformerEncoder

from .linformer_sentence_encoder_layer import LinformerTransformerEncoderLayer


class LinformerTransformerEncoder(TransformerEncoder):
    """
    Implementation for a Bi-directional Linformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    LinformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.compress_layer = None
        super().__init__(args, dictionary, embed_tokens)

    def build_encoder_layer(self, args, is_moe_layer=False):
        if self.args.shared_layer_kv_compressed == 1 and self.compress_layer is None:
            compress_layer = nn.Linear(
                self.args.max_positions,
                self.args.max_positions // self.args.compressed,
            )
            # intialize parameters for compressed layer
            nn.init.xavier_uniform_(compress_layer.weight, gain=1 / math.sqrt(2))
            if self.args.freeze_compress == 1:
                compress_layer.weight.requires_grad = False
            self.compress_layer = compress_layer

        return LinformerTransformerEncoderLayer(args, self.compress_layer)
