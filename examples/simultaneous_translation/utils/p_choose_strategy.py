from typing import Optional, Dict
from torch import Tensor
import torch


def waitk_p_choose(
    tgt_len: int,
    src_len: int,
    bsz: int,
    waitk_lagging: int,
    key_padding_mask: Optional[Tensor] = None,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None
):

    max_src_len = src_len
    if incremental_state is not None:
        # Retrieve target length from incremental states
        # For inference the length of query is always 1
        max_tgt_len = incremental_state["steps"]["tgt"]
        assert max_tgt_len is not None
        max_tgt_len = int(max_tgt_len)
    else:
        max_tgt_len = tgt_len

    if max_src_len < waitk_lagging:
        if incremental_state is not None:
            max_tgt_len = 1
        return torch.zeros(
            bsz, max_tgt_len, max_src_len
        )

    # Assuming the p_choose looks like this for wait k=3
    # src_len = 6, max_tgt_len = 5
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
    # n from 0 to max_tgt_len - 1
    #
    # First, generate the indices (activate_indices_offset: bsz, max_tgt_len)
    # Second, scatter a zeros tensor (bsz, max_tgt_len * src_len)
    # with activate_indices_offset
    # Third, resize the tensor to (bsz, max_tgt_len, src_len)

    activate_indices_offset = (
        (
            torch.arange(max_tgt_len) * (max_src_len + 1)
            + waitk_lagging - 1
        )
        .unsqueeze(0)
        .expand(bsz, max_tgt_len)
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
                    max_tgt_len,
                    max_src_len - waitk_lagging + 1
                ]
            ) * max_src_len - 1
        )
    )

    p_choose = torch.zeros(bsz, max_tgt_len * max_src_len)

    p_choose = p_choose.scatter(
        1,
        activate_indices_offset,
        1.0
    ).view(bsz, max_tgt_len, max_src_len)

    if key_padding_mask is not None:
        p_choose = p_choose.to(key_padding_mask)
        p_choose = p_choose.masked_fill(key_padding_mask.unsqueeze(1), 0)

    if incremental_state is not None:
        p_choose = p_choose[:, -1:]

    return p_choose.float()


def learnable_p_choose(
    energy,
    noise_mean: float = 0.0,
    noise_var: float = 0.0,
    training: bool = True
):
    """
    Calculating step wise prob for reading and writing
    1 to read, 0 to write
    energy: bsz, tgt_len, src_len
    """

    noise = 0
    if training:
        # add noise here to encourage discretness
        noise = (
            torch.normal(noise_mean, noise_var, energy.size())
            .type_as(energy)
            .to(energy.device)
        )

    p_choose = torch.sigmoid(energy + noise)

    # p_choose: bsz * self.num_heads, tgt_len, src_len
    return p_choose
