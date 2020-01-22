#!/usr/bin/env python3

import unittest

import torch
from fairseq.modules import multihead_attention, sinusoidal_positional_embedding


class TestExportModels(unittest.TestCase):
    def test_export_multihead_attention(self):
        module = multihead_attention.MultiheadAttention(embed_dim=8, num_heads=2)
        torch.jit.script(module)

    def test_positional_embedding(self):
        module = sinusoidal_positional_embedding.SinusoidalPositionalEmbedding(
            embedding_dim=8, padding_idx=1
        )
        torch.jit.script(module)
