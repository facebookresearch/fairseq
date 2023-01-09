from typing import Optional, Dict
from torch import Tensor
import torch


def waitk_p_choose(
    tgt_len: int,
    src_len: int,
    bsz: int,
    waitk_lagging: int,
    key_padding_mask: Optional[Tensor] = None,
    consecutive_writes: Optional[int] = 1,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
):
    max_src_len = src_len
    if incremental_state is not None:
        # Retrieve target length from incremental states
        # For inference the length of query is always 1
        if (
            "online" not in incremental_state
            or not incremental_state["online"]["only"].item()
        ):
            # not in online mode (finish reading source side input)
            p_choose = torch.zeros(bsz, 1, max_src_len)
            p_choose[:, :, -1] = 0.0
            return p_choose

        max_tgt_len = incremental_state["steps"]["tgt"]
        assert max_tgt_len is not None
        max_tgt_len = int(max_tgt_len)
    else:
        max_tgt_len = tgt_len

    if max_src_len < waitk_lagging:
        if incremental_state is not None:
            max_tgt_len = 1
        return torch.zeros(bsz, max_tgt_len, max_src_len)

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

    # If with consecutive writes
    # src_len = 5, max_tgt_len = 11, L=3
    #   [0, 0, 1, 0, 0]
    #   [0, 0, 1, 0, 0]
    #   [0, 0, 1, 0, 0]
    #   [0, 0, 0, 1, 0]
    #   [0, 0, 0, 1, 0]
    #   [0, 0, 0, 1, 0]
    #   [0, 0, 0, 0, 1]
    #   [0, 0, 0, 0, 1]
    #   [0, 0, 0, 0, 1]
    #   [0, 0, 0, 0, 1]
    #   [0, 0, 0, 0, 1]
    #   [0, 0, 0, 0, 1]
    # linearize the p_choose matrix:
    # The indices of linearized matrix that equals 1 is
    # 2 + 5 * 0
    # 2 + 5 * 1
    # 2 + 5 * 2
    # 3 + 5 * 3
    # 3 + 5 * 4
    # 3 + 5 * 5
    # 4 + 5 * 6 ...
    # floor(n / L) + src_len * n + k - 1
    # n from 0 to max_tgt_len - 1
    # l from 0 to L=consecutive_writes - 1

    activate_indices_offset = (
        (
            torch.arange(max_tgt_len) * max_src_len
            + torch.arange(max_tgt_len) / consecutive_writes
            + waitk_lagging
            - 1
        )
        .unsqueeze(0)
        .expand(bsz, max_tgt_len)
        .floor()
        .long()
    )

    if key_padding_mask is not None:
        if key_padding_mask[:, 0].any():
            # Left padding
            activate_indices_offset += key_padding_mask.sum(dim=1, keepdim=True)

    # Need to remove the indices that are too large during training
    activate_indices_offset = activate_indices_offset[
        activate_indices_offset
        < max_src_len
        * min(max_tgt_len, (max_src_len - waitk_lagging + 1) * consecutive_writes)
    ].view(bsz, -1)

    p_choose = torch.zeros(bsz, max_tgt_len * max_src_len)

    p_choose = p_choose.scatter(1, activate_indices_offset, 1.0).view(
        bsz, max_tgt_len, max_src_len
    )

    if incremental_state is not None:
        p_choose = p_choose[:, -1:]
    else:
        if key_padding_mask is not None:
            p_choose = p_choose.to(key_padding_mask)
            p_choose = p_choose.masked_fill(key_padding_mask.unsqueeze(1), 0)
        else:
            max_tgt_index = (activate_indices_offset[0, -1] / max_src_len).long() + 1
            p_choose[:, max_tgt_index:, max_src_len - 1 :] = 1

    return p_choose.float()


def learnable_p_choose(
    energy, noise_mean: float = 0.0, noise_var: float = 0.0, training: bool = True
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
