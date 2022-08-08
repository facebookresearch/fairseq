from typing import Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
import fairseq
from torch import Tensor
from fairseq import utils

# positional embeddingに倣ってファイルを分けると読みづらいので、LearnedSegmentEmbeddingもこのファイルに書く


def SegmentEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    dictionary: fairseq.data.dictionary.Dictionary,
):
    if padding_idx is not None:
        num_embeddings = num_embeddings + padding_idx + 1  # ？
    m = LearnedSegmentEmbedding(num_embeddings, embedding_dim, padding_idx, dictionary)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class LearnedSegmentEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        dictionary: fairseq.data.dictionary.Dictionary,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.sep1_idx = dictionary.indices.get('<c1>')
        self.sep2_idx = dictionary.indices.get('<c2>')
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        positions = utils.make_segments(
            input, self.padding_idx, self.sep1_idx, self.sep2_idx, onnx_trace=self.onnx_trace
        )
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
