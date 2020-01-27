#!/usr/bin/env python3

import unittest

import torch
from fairseq.modules import multihead_attention, sinusoidal_positional_embedding


class TestExportModels(unittest.TestCase):

    def test_export_multihead_attention(self):
        module = multihead_attention.MultiheadAttention(embed_dim=8, num_heads=2)
        torch.jit.script(module)

    def test_incremental_state_multihead_attention(self):
        module1 = multihead_attention.MultiheadAttention(embed_dim=8, num_heads=2)
        module1 = torch.jit.script(module1)
        module2 = multihead_attention.MultiheadAttention(embed_dim=8, num_heads=2)
        module2 = torch.jit.script(module2)

        state = {}
        state = module1.set_incremental_state(state, 'key', {'a': torch.tensor([1])})
        state = module2.set_incremental_state(state, 'key', {'a': torch.tensor([2])})
        v1 = module1.get_incremental_state(state, 'key')['a']
        v2 = module2.get_incremental_state(state, 'key')['a']

        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)

    def test_positional_embedding(self):
        module = sinusoidal_positional_embedding.SinusoidalPositionalEmbedding(
            embedding_dim=8, padding_idx=1
        )
        torch.jit.script(module)
