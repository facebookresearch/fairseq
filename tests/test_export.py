#!/usr/bin/env python3

import unittest

import torch
from fairseq.modules import multihead_attention


class TestExportModels(unittest.TestCase):
    def test_export_multihead_attention(self):
        module = multihead_attention.MultiheadAttention(embed_dim=8, num_heads=2)
        torch.jit.script(module)
