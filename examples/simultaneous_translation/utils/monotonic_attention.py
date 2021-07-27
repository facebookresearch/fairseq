from typing import Optional
import torch
from torch import Tensor

from examples.simultaneous_translation.utils.functions import (
    exclusive_cumprod,
    prob_check,
    moving_sum,
)


def expected_alignment_from_p_choose(
    p_choose: Tensor,
    padding_mask: Optional[Tensor] = None,
    eps: float = 1e-6
):
    """
    Calculating expected alignment for from stepwise probability

    Reference:
    Online and Linear-Time Attention by Enforcing Monotonic Alignments
    https://arxiv.org/pdf/1704.00784.pdf

    q_ij = (1 − p_{ij−1})q_{ij−1} + a+{i−1j}
    a_ij = p_ij q_ij

    Parallel solution:
    ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))

    ============================================================
    Expected input size
    p_choose: bsz, tgt_len, src_len
    """
    prob_check(p_choose)

    # p_choose: bsz, tgt_len, src_len
    bsz, tgt_len, src_len = p_choose.size()
    dtype = p_choose.dtype

    p_choose = p_choose.float()

    if padding_mask is not None:
        p_choose = p_choose.masked_fill(padding_mask.unsqueeze(1), 0.0)

    # cumprod_1mp : bsz, tgt_len, src_len
    cumprod_1mp = exclusive_cumprod(1 - p_choose, dim=2, eps=eps)
    cumprod_1mp_clamp = torch.clamp(cumprod_1mp, eps, 1.0)

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

    # Mix precision to prevent overflow for fp16
    alpha = alpha.type(dtype)

    prob_check(alpha)

    return alpha


def expected_soft_attention(
    alpha: Tensor,
    soft_energy: Tensor,
    padding_mask: Optional[Tensor] = None,
    chunk_size: Optional[int] = None,
    eps: float = 1e-10
):
    """
    Function to compute expected soft attention for
    monotonic infinite lookback attention from
    expected alignment and soft energy.

    Reference:
    Monotonic Chunkwise Attention
    https://arxiv.org/abs/1712.05382

    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    https://arxiv.org/abs/1906.05218

    alpha: bsz, tgt_len, src_len
    soft_energy: bsz, tgt_len, src_len
    padding_mask: bsz, src_len
    left_padding: bool
    """
    if padding_mask is not None:
        alpha = alpha.masked_fill(padding_mask.unsqueeze(1), 0.0)
        soft_energy = soft_energy.masked_fill(
            padding_mask.unsqueeze(1), -float("inf")
        )

    prob_check(alpha)

    dtype = alpha.dtype

    alpha = alpha.float()
    soft_energy = soft_energy.float()

    soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
    exp_soft_energy = torch.exp(soft_energy) + eps

    if chunk_size is not None:
        # Chunkwise
        beta = (
            exp_soft_energy
            * moving_sum(
                alpha / (eps + moving_sum(exp_soft_energy, chunk_size, 1)),
                1, chunk_size
            )
        )
    else:
        # Infinite lookback
        # Notice that infinite lookback is a special case of chunkwise
        # where chunksize = inf
        inner_items = alpha / (eps + torch.cumsum(exp_soft_energy, dim=2))

        beta = (
            exp_soft_energy
            * torch.cumsum(inner_items.flip(dims=[2]), dim=2)
            .flip(dims=[2])
        )

    if padding_mask is not None:
        beta = beta.masked_fill(
            padding_mask.unsqueeze(1).to(torch.bool), 0.0)

    # Mix precision to prevent overflow for fp16
    beta = beta.type(dtype)

    prob_check(beta)

    return beta


def mass_preservation(
    alpha: Tensor,
    padding_mask: Optional[Tensor] = None,
    left_padding: bool = False
):
    """
    Function to compute the mass perservation for alpha.
    This means that the residual weights of alpha will be assigned
    to the last token.

    Reference:
    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    https://arxiv.org/abs/1906.05218

    alpha: bsz, tgt_len, src_len
    padding_mask: bsz, src_len
    left_padding: bool
    """

    prob_check(alpha)

    if padding_mask is not None:
        if not left_padding:
            assert not padding_mask[:, 0].any(), (
                "Find padding on the beginning of the sequence."
            )
        alpha = alpha.masked_fill(padding_mask.unsqueeze(1), 0.0)

    if left_padding or padding_mask is None:
        residuals = 1 - alpha[:, :, :-1].sum(dim=-1).clamp(0, 1)
        alpha[:, :, -1] = residuals
    else:
        # right padding
        _, tgt_len, src_len = alpha.size()
        residuals = 1 - alpha.sum(dim=-1, keepdim=True).clamp(0, 1)
        src_lens = src_len - padding_mask.sum(dim=1, keepdim=True)
        src_lens = src_lens.expand(-1, tgt_len).contiguous()
        # add back the last value
        residuals += alpha.gather(2, src_lens.unsqueeze(2) - 1)
        alpha = alpha.scatter(2, src_lens.unsqueeze(2) - 1, residuals)

        prob_check(alpha)

    return alpha
