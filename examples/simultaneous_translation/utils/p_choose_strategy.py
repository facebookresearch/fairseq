from typing import Optional, Dict
from torch import Tensor
import torch


def waitk(
    query, key, waitk_lagging: int, num_heads: int, key_padding_mask: Optional[Tensor] = None,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None
):
    if incremental_state is not None:
        # Retrieve target length from incremental states
        # For inference the length of query is always 1
        tgt_len = incremental_state["steps"]["tgt"]
        assert tgt_len is not None
        tgt_len = int(tgt_len)
    else:
        tgt_len, bsz, _ = query.size()

    max_src_len, bsz, _ = key.size()

    if max_src_len < waitk_lagging:
        if incremental_state is not None:
            tgt_len = 1
        return query.new_zeros(
            bsz * num_heads, tgt_len, max_src_len
        )

    # Assuming the p_choose looks like this for wait k=3
    # src_len = 6, tgt_len = 5
    #   [0, 0, 1, 0, 0, 0, 0]
    #   [0, 0, 0, 1, 0, 0, 0]
    #   [0, 0, 0, 0, 1, 0, 0]
    #   [0, 0, 0, 0, 0, 1, 0]
    #   [0, 0, 0, 0, 0, 0, 1]
    # linearize the p_choose matrix:
    # [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0...]
    # The indices of linearized matrix that equals 1 is
    # 2 + 6 * 0
    # 3 + 6 * 1
    # ...
    # n + src_len * n + k - 1 = n * (src_len + 1) + k - 1
    # n from 0 to tgt_len - 1
    #
    # First, generate the indices (activate_indices_offset: bsz, tgt_len)
    # Second, scatter a zeros tensor (bsz, tgt_len * src_len)
    # with activate_indices_offset
    # Third, resize the tensor to (bsz, tgt_len, src_len)

    activate_indices_offset = (
        (
            torch.arange(tgt_len) * (max_src_len + 1)
            + waitk_lagging - 1
        )
        .unsqueeze(0)
        .expand(bsz, tgt_len)
        .to(query)
        .long()
    )

    if key_padding_mask is not None:
        if key_padding_mask[:, 0].any():
            # Left padding
            activate_indices_offset += (
                key_padding_mask.sum(dim=1, keepdim=True)
            )

    # Need to clamp the indices that are too large
    activate_indices_offset = (
        activate_indices_offset
        .clamp(
            0,
            min(
                [
                    tgt_len,
                    max_src_len - waitk_lagging + 1
                ]
            ) * max_src_len - 1
        )
    )

    p_choose = torch.zeros(bsz, tgt_len * max_src_len).to(query)

    p_choose = p_choose.scatter(
        1,
        activate_indices_offset,
        1.0
    ).view(bsz, tgt_len, max_src_len)

    if incremental_state is not None:
        p_choose = p_choose[:, -1:]
        tgt_len = 1

    # Extend to each head
    p_choose = (
        p_choose.contiguous()
        .unsqueeze(1)
        .expand(-1, num_heads, -1, -1)
        .contiguous()
        .view(-1, tgt_len, max_src_len)
    )

    return p_choose


def hard_aligned(q_proj: Optional[Tensor], k_proj: Optional[Tensor], attn_energy, noise_mean: float = 0.0, noise_var: float = 0.0, training: bool = True):
    """
    Calculating step wise prob for reading and writing
    1 to read, 0 to write
    """

    noise = 0
    if training:
        # add noise here to encourage discretness
        noise = (
            torch.normal(noise_mean, noise_var, attn_energy.size())
            .type_as(attn_energy)
            .to(attn_energy.device)
        )

    p_choose = torch.sigmoid(attn_energy + noise)
    _, _, tgt_len, src_len = p_choose.size()

    # p_choose: bsz * self.num_heads, tgt_len, src_len
    return p_choose.view(-1, tgt_len, src_len)
