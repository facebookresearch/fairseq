import math
import torch

def get_alibi(
    max_positions: int,
    attention_heads: int,
):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))
    # prepare alibi position linear bias. Note that wav2vec2 is non
    # autoregressive model so we want a symmetric mask with 0 on the
    # diagonal and other wise linear decreasing valuees
    pos_bias = (
        torch.abs(
            torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
        )
        * -1
    )
    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )
    return alibi_bias

def masked_alibi(alibi_bias, mask_indices, orig_B, orig_T):
    alibi_bias = alibi_bias.view(orig_B, -1, orig_T, orig_T)
    H = alibi_bias.size(1)
    alibi_mask = mask_indices.unsqueeze(1)
    alibi_bias = alibi_bias.masked_select(alibi_mask.unsqueeze(-1))
    alibi_bias = alibi_bias.view(orig_B, H, -1, orig_T)
    M = alibi_bias.size(-2)
    alibi_bias = alibi_bias.masked_select(alibi_mask.unsqueeze(-2))
    alibi_bias = alibi_bias.view(-1, M, M)
    return alibi_bias


