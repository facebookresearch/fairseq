from functools import partial
import torch
import torch.nn.functional as F
from .monotonic_multihead_attention import (
    MonotonicMultiheadAttentionWaitk,
    MonotonicMultiheadAttentionHard,
    MonotonicMultiheadAttentionInfiniteLookback
)
from . import register_monotonic_attention


def fixed_pooling_monotonic_attention(monotonic_attention):
    def create_model(monotonic_attention, klass):
        class FixedStrideMonotonicAttention(monotonic_attention):
            def __init__(self, args):
                super().__init__(args)
                self.pooling_type = args.fixed_pooling_type
                self.pooling_ratio = args.fixed_pooling_ratio
                self.pooling_pad_threshold = args.fixed_pooling_pad_threshold
                if self.pooling_ratio == 1:
                    return

                # TODO: A learnable layer, e.g. CNN layer?
                if args.fixed_pooling_type == "average":
                    self.pooling_layer = torch.nn.AvgPool1d(
                        kernel_size=self.pooling_ratio,
                        stride=self.pooling_ratio,
                        ceil_mode=True
                    )
                elif args.fixed_pooling_type == "last":
                    def last(key):
                        if key.size(2) < self.pooling_ratio:
                            return key
                        else:
                            k = key[
                                :, :,
                                self.pooling_ratio - 1::self.pooling_ratio
                            ].contiguous()
                            if key.size(-1) % self.pooling_ratio != 0:
                                k = torch.cat(
                                    [k, key[:, :, -1:]],
                                    dim=-1
                                ).contiguous()
                            return k
                    self.pooling_layer = last
                else:
                    raise NotImplementedError

            @staticmethod
            def add_args(parser):
                super(
                    FixedStrideMonotonicAttention,
                    FixedStrideMonotonicAttention
                ).add_args(parser)

                parser.add_argument("--fixed-pooling-type", default="average",
                                    help="Pooling type")
                parser.add_argument("--fixed-pooling-ratio",
                                    type=int, required=True,
                                    help="ratio for pooling")
                parser.add_argument("--fixed-pooling-pad-threshold",
                                    type=float, default=0.3,
                                    help="If a part of the sequence has pad"
                                    ",the threshold the pooled part is a pad.")

            def insert_zeros(self, x):
                bsz_num_heads, tgt_len, src_len = x.size()
                stride = self.pooling_ratio
                weight = F.pad(x.new_ones(1, 1, 1), (stride - 1, 0))
                x_upsample = F.conv_transpose1d(
                    x.view(-1, src_len).unsqueeze(1),
                    weight, stride=stride, padding=0,
                )
                return x_upsample.squeeze(1).view(bsz_num_heads, tgt_len, -1)

            def p_choose(
                self, query, key, key_padding_mask=None,
                incremental_state=None, **extra_args
            ):

                if self.pooling_ratio == 1:
                    return super().p_choose(
                        self, query, key, key_padding_mask=None,
                        incremental_state=None, **extra_args
                    )

                key_pool = self.pooling_layer(
                    key.transpose(0, 2)
                ).transpose(0, 2)

                if key_padding_mask is not None:
                    key_padding_mask_pool = self.pooling_layer(
                        key_padding_mask.unsqueeze(0).float()
                    ).squeeze(0).gt(self.pooling_pad_threshold)
                    # Make sure at least one element is not pad
                    key_padding_mask_pool[:, 0] = 0
                else:
                    key_padding_mask_pool = None

                p_choose_pooled = super().p_choose(
                    query, key_pool, key_padding_mask_pool,
                    incremental_state=incremental_state,
                )

                # Upsample, interpolate zeros
                p_choose = self.insert_zeros(p_choose_pooled)

                # can be larger than src_len because we used ceil before
                src_len = key.size(0)
                p_choose = p_choose[:, :, :src_len]
                p_choose[:, :, -1] = p_choose_pooled[:, :, -1]

                tgt_len = query.size(0)
                batch_size = query.size(1)

                assert list(p_choose.size()) == [
                    batch_size * self.num_heads, tgt_len, src_len]

                return p_choose

        FixedStrideMonotonicAttention.__name__ = klass.__name__
        return FixedStrideMonotonicAttention
    return partial(create_model, monotonic_attention)


@register_monotonic_attention("waitk_fixed_pooling")
@fixed_pooling_monotonic_attention(MonotonicMultiheadAttentionWaitk)
class MonotonicMultiheadAttentionWaitkFixedStride:
    pass


@register_monotonic_attention("hard_aligned_fixed_pooling")
@fixed_pooling_monotonic_attention(MonotonicMultiheadAttentionHard)
class MonotonicMultiheadAttentionHardFixedStride:
    pass


@register_monotonic_attention("infinite_lookback_fixed_pooling")
@fixed_pooling_monotonic_attention(MonotonicMultiheadAttentionInfiniteLookback)
class MonotonicMultiheadAttentionInfiniteLookbackFixedStride:
    pass
