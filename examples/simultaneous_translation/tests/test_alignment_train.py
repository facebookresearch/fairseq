import unittest

import numpy as np
import torch

import hypothesis.strategies as st
from hypothesis import assume, given, settings
from torch.testing._internal.common_utils import TestCase
from examples.simultaneous_translation.utils.functions import exclusive_cumprod


TEST_CUDA = torch.cuda.is_available()


class AlignmentTrainTest(TestCase):
    def _test_custom_alignment_train_ref(self, p_choose, eps):
        cumprod_1mp = exclusive_cumprod(1 - p_choose, dim=2, eps=eps)
        cumprod_1mp_clamp = torch.clamp(cumprod_1mp, eps, 1.0)

        bsz = p_choose.size(0)
        tgt_len = p_choose.size(1)
        src_len = p_choose.size(2)

        alpha_0 = p_choose.new_zeros([bsz, 1, src_len])
        alpha_0[:, :, 0] = 1.0

        previous_alpha = [alpha_0]

        for i in range(tgt_len):
            # p_choose: bsz , tgt_len, src_len
            # cumprod_1mp_clamp : bsz, tgt_len, src_len
            # previous_alpha[i]: bsz, 1, src_len
            # alpha_i: bsz, src_len
            alpha_i = (
                p_choose[:, i]
                * cumprod_1mp[:, i]
                * torch.cumsum(
                    previous_alpha[i][:, 0] / cumprod_1mp_clamp[:, i], dim=1
                )
            ).clamp(0, 1.0)

            previous_alpha.append(alpha_i.unsqueeze(1))

        # alpha: bsz * num_heads, tgt_len, src_len
        alpha = torch.cat(previous_alpha[1:], dim=1)
        return alpha

    def _test_custom_alignment_train_impl(self, p_choose, alpha, eps):
        if p_choose.is_cuda:
            from alignment_train_cuda_binding import alignment_train_cuda  # @manual=//deeplearning/projects/fairseq-py:alignment_train_cuda_binding
            alignment_train_cuda(p_choose, alpha, eps)
        else:
            from alignment_train_cpu_binding import alignment_train_cpu  # @manual=//deeplearning/projects/fairseq-py:alignment_train_cpu_binding
            alignment_train_cpu(p_choose, alpha, eps)

    @settings(deadline=None)
    @given(
        bsz=st.integers(1, 100),
        tgt_len=st.integers(1, 100),
        src_len=st.integers(1, 550),
        device=st.sampled_from(["cpu", "cuda"]),
    )
    def test_alignment_train(self, bsz, tgt_len, src_len, device):
        eps = 1e-6

        assume(device == "cpu" or TEST_CUDA)
        p_choose = torch.rand(bsz, tgt_len, src_len, device=device)

        # run the alignment with the custom operator
        alpha_act = p_choose.new_zeros([bsz, tgt_len, src_len])
        self._test_custom_alignment_train_impl(p_choose, alpha_act, eps)

        # runu the alignment with the ref implementation
        alpha_ref = self._test_custom_alignment_train_ref(p_choose, eps)

        # verify the results
        alpha_act = alpha_act.cpu().detach().numpy()
        alpha_ref = alpha_ref.cpu().detach().numpy()
        np.testing.assert_allclose(
            alpha_act,
            alpha_ref,
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
