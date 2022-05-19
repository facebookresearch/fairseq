import torch
import numpy as np
import unittest
from fairseq.modules import (
    ESPNETMultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
)

torch.use_deterministic_algorithms(True)


class TestESPNETMultiHeadedAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 3
        self.B = 1
        self.C = 2
        torch.manual_seed(0)
        self.sample = torch.randn(self.T, self.B, self.C)  # TBC
        self.sample_scores = torch.randn(self.B, 1, self.T, self.T)
        self.MHA = ESPNETMultiHeadedAttention(self.C, 1, dropout=0)

    def test_forward(self):
        expected_scores = torch.tensor(
            [[[0.1713, -0.3776]], [[0.2263, -0.4486]], [[0.2243, -0.4538]]]
        )
        scores, _ = self.MHA(self.sample, self.sample, self.sample)
        self.assertTrue(
            np.allclose(
                expected_scores.cpu().detach().numpy(),
                scores.cpu().detach().numpy(),
                atol=1e-4,
            )
        )

    def test_forward_qkv(self):
        expected_query = torch.tensor(
            [[[[-1.0235, 0.0409], [0.4008, 1.3077], [0.5396, 2.0698]]]]
        )
        expected_key = torch.tensor(
            [[[[0.5053, -0.4965], [-0.3730, -0.9473], [-0.7019, -0.1935]]]]
        )
        expected_val = torch.tensor(
            [[[[-0.9940, 0.5403], [0.5924, -0.7619], [0.7504, -1.0892]]]]
        )
        sample_t = self.sample.transpose(0, 1)
        query, key, val = self.MHA.forward_qkv(sample_t, sample_t, sample_t)
        self.assertTrue(
            np.allclose(
                expected_query.cpu().detach().numpy(),
                query.cpu().detach().numpy(),
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_key.cpu().detach().numpy(),
                key.cpu().detach().numpy(),
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_val.cpu().detach().numpy(),
                val.cpu().detach().numpy(),
                atol=1e-4,
            )
        )

    def test_forward_attention(self):
        expected_scores = torch.tensor(
            [[[0.1627, -0.6249], [-0.2547, -0.6487], [-0.0711, -0.8545]]]
        )
        scores = self.MHA.forward_attention(
            self.sample.transpose(0, 1).view(self.B, 1, self.T, self.C),
            self.sample_scores,
            mask=None,
        )
        self.assertTrue(
            np.allclose(
                expected_scores.cpu().detach().numpy(),
                scores.cpu().detach().numpy(),
                atol=1e-4,
            )
        )


class TestRelPositionMultiHeadedAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 3
        self.B = 1
        self.C = 2
        torch.manual_seed(0)
        self.sample = torch.randn(self.T, self.B, self.C)  # TBC
        self.sample_x = torch.randn(self.B, 1, self.T, self.T * 2 - 1)
        self.sample_pos = torch.randn(self.B, self.T * 2 - 1, self.C)
        self.MHA = RelPositionMultiHeadedAttention(self.C, 1, dropout=0)

    def test_rel_shift(self):
        expected_x = torch.tensor(
            [
                [
                    [
                        [-0.7193, -0.4033, -0.5966],
                        [-0.8567, 1.1006, -1.0712],
                        [-0.5663, 0.3731, -0.8920],
                    ]
                ]
            ]
        )
        x = self.MHA.rel_shift(self.sample_x)
        self.assertTrue(
            np.allclose(
                expected_x.cpu().detach().numpy(),
                x.cpu().detach().numpy(),
                atol=1e-4,
            )
        )

    def test_forward(self):
        expected_scores = torch.tensor(
            [
                [[-0.9609, -0.5020]],
                [[-0.9308, -0.4890]],
                [[-0.9473, -0.4948]],
                [[-0.9609, -0.5020]],
                [[-0.9308, -0.4890]],
                [[-0.9473, -0.4948]],
                [[-0.9609, -0.5020]],
                [[-0.9308, -0.4890]],
                [[-0.9473, -0.4948]],
                [[-0.9609, -0.5020]],
                [[-0.9308, -0.4890]],
                [[-0.9473, -0.4948]],
                [[-0.9609, -0.5020]],
                [[-0.9308, -0.4890]],
                [[-0.9473, -0.4948]],
            ]
        )
        scores, _ = self.MHA(self.sample, self.sample, self.sample, self.sample_pos)
        self.assertTrue(
            np.allclose(
                expected_scores.cpu().detach().numpy(),
                scores.cpu().detach().numpy(),
                atol=1e-4,
            )
        )


class TestRotaryPositionMultiHeadedAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 3
        self.B = 1
        self.C = 2
        torch.manual_seed(0)
        self.sample = torch.randn(self.T, self.B, self.C)  # TBC
        self.MHA = RotaryPositionMultiHeadedAttention(
            self.C, 1, dropout=0, precision=None
        )

    def test_forward(self):
        expected_scores = torch.tensor(
            [[[-0.3220, -0.4726]], [[-1.2813, -0.0979]], [[-0.3138, -0.4758]]]
        )
        scores, _ = self.MHA(self.sample, self.sample, self.sample)
        self.assertTrue(
            np.allclose(
                expected_scores.cpu().detach().numpy(),
                scores.cpu().detach().numpy(),
                atol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()
