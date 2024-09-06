import torch
import numpy as np
import unittest
from fairseq.modules.rotary_positional_embedding import apply_rotary_pos_emb
from fairseq.modules import RotaryPositionalEmbedding


class TestRotaryPositionalEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 3
        self.B = 1
        self.C = 2
        torch.manual_seed(0)
        self.sample = torch.randn(self.T, self.B, self.C)  # TBC
        self.rope_pos_emd = RotaryPositionalEmbedding(dim=self.C)

    def test_forward(self):
        expected_cos = torch.tensor(
            [[[[1.0000, 1.0000]]], [[[0.5403, 0.5403]]], [[[-0.4161, -0.4161]]]]
        )
        expected_sin = torch.tensor(
            [[[[0.0000, 0.0000]]], [[[0.8415, 0.8415]]], [[[0.9093, 0.9093]]]]
        )
        cos, sin = self.rope_pos_emd(self.sample, self.T)
        self.assertTrue(
            np.allclose(
                expected_cos.cpu().detach().numpy(),
                cos.cpu().detach().numpy(),
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_sin.cpu().detach().numpy(),
                sin.cpu().detach().numpy(),
                atol=1e-4,
            )
        )

    def test_apply_rotary_pos_emb(self):
        cos, sin = self.rope_pos_emd(self.sample, self.T)
        query = self.sample.view(self.T, self.B, 1, self.C)
        expected_query = torch.tensor(
            [[[[1.5410, -0.2934]]], [[[-1.6555, -1.5263]]], [[[1.7231, -0.4041]]]]
        )
        new_query, new_key = apply_rotary_pos_emb(query, query, cos, sin)
        self.assertTrue(
            np.allclose(
                expected_query.cpu().detach().numpy(),
                new_query.cpu().detach().numpy(),
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_query.cpu().detach().numpy(),
                new_key.cpu().detach().numpy(),
                atol=1e-4,
            )
        )

    def test_jit_compile_rope_module(self):
        module_scripted = torch.jit.script(self.rope_pos_emd)
        apply_rotary_scripted = torch.jit.script(apply_rotary_pos_emb)
        # Test several different lengths
        for T in [3, 5, 10]:
            sample = torch.randn(T, self.B, self.C)
            # Run forward pass with the original module
            cos_original, sin_original = self.rope_pos_emd(sample, T)
            query = sample.view(T, self.B, 1, self.C)
            new_query, new_key = apply_rotary_pos_emb(query, query, cos_original, sin_original)

            # Run forward pass with the scripted module
            cos_scripted, sin_scripted = module_scripted(sample, T)
            new_query_scripted, new_key_scripted = apply_rotary_scripted(query, query, cos_scripted, sin_scripted)

            # Ensure the outputs are the same
            self.assertTrue(torch.allclose(cos_original, cos_scripted))
            self.assertTrue(torch.allclose(sin_original, sin_scripted))
            self.assertTrue(torch.allclose(new_query, new_query_scripted))
            self.assertTrue(torch.allclose(new_key, new_key_scripted))


if __name__ == "__main__":
    unittest.main()
