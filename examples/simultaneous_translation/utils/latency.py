# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class LatencyMetric(object):
    @staticmethod
    def length_from_padding_mask(padding_mask, batch_first: bool = False):
        dim = 1 if batch_first else 0
        return padding_mask.size(dim) - padding_mask.sum(dim=dim, keepdim=True)

    def prepare_latency_metric(
        self,
        delays,
        src_lens,
        target_padding_mask=None,
        batch_first: bool = False,
        start_from_zero: bool = True,
    ):
        assert len(delays.size()) == 2
        assert len(src_lens.size()) == 2

        if start_from_zero:
            delays = delays + 1

        if batch_first:
            # convert to batch_last
            delays = delays.t()
            src_lens = src_lens.t()
            tgt_len, bsz = delays.size()
            _, bsz_1 = src_lens.size()

            if target_padding_mask is not None:
                target_padding_mask = target_padding_mask.t()
                tgt_len_1, bsz_2 = target_padding_mask.size()
                assert tgt_len == tgt_len_1
                assert bsz == bsz_2

        assert bsz == bsz_1

        if target_padding_mask is None:
            tgt_lens = tgt_len * delays.new_ones([1, bsz]).float()
        else:
            # 1, batch_size
            tgt_lens = self.length_from_padding_mask(target_padding_mask, False).float()
            delays = delays.masked_fill(target_padding_mask, 0)

        return delays, src_lens, tgt_lens, target_padding_mask

    def __call__(
        self,
        delays,
        src_lens,
        target_padding_mask=None,
        batch_first: bool = False,
        start_from_zero: bool = True,
    ):
        delays, src_lens, tgt_lens, target_padding_mask = self.prepare_latency_metric(
            delays, src_lens, target_padding_mask, batch_first, start_from_zero
        )
        return self.cal_metric(delays, src_lens, tgt_lens, target_padding_mask)

    @staticmethod
    def cal_metric(delays, src_lens, tgt_lens, target_padding_mask):
        """
        Expected sizes:
        delays: tgt_len, batch_size
        src_lens: 1, batch_size
        target_padding_mask: tgt_len, batch_size
        """
        raise NotImplementedError


class AverageProportion(LatencyMetric):
    """
    Function to calculate Average Proportion from
    Can neural machine translation do simultaneous translation?
    (https://arxiv.org/abs/1606.02012)

    Delays are monotonic steps, range from 1 to src_len.
    Give src x tgt y, AP is calculated as:

    AP = 1 / (|x||y]) sum_i^|Y| deleys_i
    """

    @staticmethod
    def cal_metric(delays, src_lens, tgt_lens, target_padding_mask):
        if target_padding_mask is not None:
            AP = torch.sum(
                delays.masked_fill(target_padding_mask, 0), dim=0, keepdim=True
            )
        else:
            AP = torch.sum(delays, dim=0, keepdim=True)

        AP = AP / (src_lens * tgt_lens)
        return AP


class AverageLagging(LatencyMetric):
    """
    Function to calculate Average Lagging from
    STACL: Simultaneous Translation with Implicit Anticipation
    and Controllable Latency using Prefix-to-Prefix Framework
    (https://arxiv.org/abs/1810.08398)

    Delays are monotonic steps, range from 1 to src_len.
    Give src x tgt y, AP is calculated as:

    AL = 1 / tau sum_i^tau delays_i - (i - 1) / gamma

    Where
    gamma = |y| / |x|
    tau = argmin_i(delays_i = |x|)
    """

    @staticmethod
    def cal_metric(delays, src_lens, tgt_lens, target_padding_mask):
        # tau = argmin_i(delays_i = |x|)
        tgt_len, bsz = delays.size()
        lagging_padding_mask = delays >= src_lens
        lagging_padding_mask = torch.nn.functional.pad(
            lagging_padding_mask.t(), (1, 0)
        ).t()[:-1, :]
        gamma = tgt_lens / src_lens
        lagging = (
            delays
            - torch.arange(delays.size(0))
            .unsqueeze(1)
            .type_as(delays)
            .expand_as(delays)
            / gamma
        )
        lagging.masked_fill_(lagging_padding_mask, 0)
        tau = (1 - lagging_padding_mask.type_as(lagging)).sum(dim=0, keepdim=True)
        AL = lagging.sum(dim=0, keepdim=True) / tau

        return AL


class DifferentiableAverageLagging(LatencyMetric):
    """
    Function to calculate Differentiable Average Lagging from
    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    (https://arxiv.org/abs/1906.05218)

    Delays are monotonic steps, range from 0 to src_len-1.
    (In the original paper thery are from 1 to src_len)
    Give src x tgt y, AP is calculated as:

    DAL = 1 / |Y| sum_i^|Y| delays'_i - (i - 1) / gamma

    Where
    delays'_i =
        1. delays_i if i == 1
        2. max(delays_i, delays'_{i-1} + 1 / gamma)

    """

    @staticmethod
    def cal_metric(delays, src_lens, tgt_lens, target_padding_mask):
        tgt_len, bsz = delays.size()

        gamma = tgt_lens / src_lens
        new_delays = torch.zeros_like(delays)

        for i in range(delays.size(0)):
            if i == 0:
                new_delays[i] = delays[i]
            else:
                new_delays[i] = torch.cat(
                    [
                        new_delays[i - 1].unsqueeze(0) + 1 / gamma,
                        delays[i].unsqueeze(0),
                    ],
                    dim=0,
                ).max(dim=0)[0]

        DAL = (
            new_delays
            - torch.arange(delays.size(0))
            .unsqueeze(1)
            .type_as(delays)
            .expand_as(delays)
            / gamma
        )
        if target_padding_mask is not None:
            DAL = DAL.masked_fill(target_padding_mask, 0)

        DAL = DAL.sum(dim=0, keepdim=True) / tgt_lens

        return DAL


class LatencyMetricVariance(LatencyMetric):
    def prepare_latency_metric(
        self,
        delays,
        src_lens,
        target_padding_mask=None,
        batch_first: bool = True,
        start_from_zero: bool = True,
    ):
        assert batch_first
        assert len(delays.size()) == 3
        assert len(src_lens.size()) == 2

        if start_from_zero:
            delays = delays + 1

        # convert to batch_last
        bsz, num_heads_x_layers, tgt_len = delays.size()
        bsz_1, _ = src_lens.size()
        assert bsz == bsz_1

        if target_padding_mask is not None:
            bsz_2, tgt_len_1 = target_padding_mask.size()
            assert tgt_len == tgt_len_1
            assert bsz == bsz_2

        if target_padding_mask is None:
            tgt_lens = tgt_len * delays.new_ones([bsz, tgt_len]).float()
        else:
            # batch_size, 1
            tgt_lens = self.length_from_padding_mask(target_padding_mask, True).float()
            delays = delays.masked_fill(target_padding_mask.unsqueeze(1), 0)

        return delays, src_lens, tgt_lens, target_padding_mask


class VarianceDelay(LatencyMetricVariance):
    @staticmethod
    def cal_metric(delays, src_lens, tgt_lens, target_padding_mask):
        """
        delays : bsz, num_heads_x_layers, tgt_len
        src_lens : bsz, 1
        target_lens : bsz, 1
        target_padding_mask: bsz, tgt_len or None
        """
        if delays.size(1) == 1:
            return delays.new_zeros([1])

        variance_delays = delays.var(dim=1)

        if target_padding_mask is not None:
            variance_delays.masked_fill_(target_padding_mask, 0)

        return variance_delays.sum(dim=1, keepdim=True) / tgt_lens


class LatencyInference(object):
    def __init__(self, start_from_zero=True):
        self.metric_calculator = {
            "differentiable_average_lagging": DifferentiableAverageLagging(),
            "average_lagging": AverageLagging(),
            "average_proportion": AverageProportion(),
        }

        self.start_from_zero = start_from_zero

    def __call__(self, monotonic_step, src_lens):
        """
        monotonic_step range from 0 to src_len. src_len means eos
        delays: bsz, tgt_len
        src_lens: bsz, 1
        """
        if not self.start_from_zero:
            monotonic_step -= 1

        src_lens = src_lens

        delays = monotonic_step.view(
            monotonic_step.size(0), -1, monotonic_step.size(-1)
        ).max(dim=1)[0]

        delays = delays.masked_fill(delays >= src_lens, 0) + (src_lens - 1).expand_as(
            delays
        ).masked_fill(delays < src_lens, 0)
        return_dict = {}
        for key, func in self.metric_calculator.items():
            return_dict[key] = func(
                delays.float(),
                src_lens.float(),
                target_padding_mask=None,
                batch_first=True,
                start_from_zero=True,
            ).t()

        return return_dict


class LatencyTraining(object):
    def __init__(
        self,
        avg_weight,
        var_weight,
        avg_type,
        var_type,
        stay_on_last_token,
        average_method,
    ):
        self.avg_weight = avg_weight
        self.var_weight = var_weight
        self.avg_type = avg_type
        self.var_type = var_type
        self.stay_on_last_token = stay_on_last_token
        self.average_method = average_method

        self.metric_calculator = {
            "differentiable_average_lagging": DifferentiableAverageLagging(),
            "average_lagging": AverageLagging(),
            "average_proportion": AverageProportion(),
        }

        self.variance_calculator = {
            "variance_delay": VarianceDelay(),
        }

    def expected_delays_from_attention(
        self, attention, source_padding_mask=None, target_padding_mask=None
    ):
        if type(attention) == list:
            # bsz, num_heads, tgt_len, src_len
            bsz, num_heads, tgt_len, src_len = attention[0].size()
            attention = torch.cat(attention, dim=1)
            bsz, num_heads_x_layers, tgt_len, src_len = attention.size()
            # bsz * num_heads * num_layers, tgt_len, src_len
            attention = attention.view(-1, tgt_len, src_len)
        else:
            # bsz * num_heads * num_layers, tgt_len, src_len
            bsz, tgt_len, src_len = attention.size()
            num_heads_x_layers = 1
            attention = attention.view(-1, tgt_len, src_len)

        if not self.stay_on_last_token:
            residual_attention = 1 - attention[:, :, :-1].sum(dim=2, keepdim=True)
            attention = torch.cat([attention[:, :, :-1], residual_attention], dim=2)

        # bsz * num_heads_x_num_layers, tgt_len, src_len for MMA
        steps = (
            torch.arange(1, 1 + src_len)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand_as(attention)
            .type_as(attention)
        )

        if source_padding_mask is not None:
            src_offset = (
                source_padding_mask.type_as(attention)
                .sum(dim=1, keepdim=True)
                .expand(bsz, num_heads_x_layers)
                .contiguous()
                .view(-1, 1)
            )
            src_lens = src_len - src_offset
            if source_padding_mask[:, 0].any():
                # Pad left
                src_offset = src_offset.view(-1, 1, 1)
                steps = steps - src_offset
                steps = steps.masked_fill(steps <= 0, 0)
        else:
            src_lens = attention.new_ones([bsz, num_heads_x_layers]) * src_len
            src_lens = src_lens.view(-1, 1)

        # bsz * num_heads_num_layers, tgt_len, src_len
        expected_delays = (
            (steps * attention).sum(dim=2).view(bsz, num_heads_x_layers, tgt_len)
        )

        if target_padding_mask is not None:
            expected_delays.masked_fill_(target_padding_mask.unsqueeze(1), 0)

        return expected_delays, src_lens

    def avg_loss(self, expected_delays, src_lens, target_padding_mask):

        bsz, num_heads_x_layers, tgt_len = expected_delays.size()
        target_padding_mask = (
            target_padding_mask.unsqueeze(1)
            .expand_as(expected_delays)
            .contiguous()
            .view(-1, tgt_len)
        )

        if self.average_method == "average":
            # bsz * tgt_len
            expected_delays = expected_delays.mean(dim=1)
        elif self.average_method == "weighted_average":
            weights = torch.nn.functional.softmax(expected_delays, dim=1)
            expected_delays = torch.sum(expected_delays * weights, dim=1)
        elif self.average_method == "max":
            # bsz * num_heads_x_num_layers, tgt_len
            expected_delays = expected_delays.max(dim=1)[0]
        else:
            raise RuntimeError(f"{self.average_method} is not supported")

        src_lens = src_lens.view(bsz, -1)[:, :1]
        target_padding_mask = target_padding_mask.view(bsz, -1, tgt_len)[:, 0]

        if self.avg_weight > 0.0:
            if self.avg_type in self.metric_calculator:
                average_delays = self.metric_calculator[self.avg_type](
                    expected_delays,
                    src_lens,
                    target_padding_mask,
                    batch_first=True,
                    start_from_zero=False,
                )
            else:
                raise RuntimeError(f"{self.avg_type} is not supported.")

            # bsz * num_heads_x_num_layers, 1
            return self.avg_weight * average_delays.sum()
        else:
            return 0.0

    def var_loss(self, expected_delays, src_lens, target_padding_mask):
        src_lens = src_lens.view(expected_delays.size(0), expected_delays.size(1))[
            :, :1
        ]
        if self.var_weight > 0.0:
            if self.var_type in self.variance_calculator:
                variance_delays = self.variance_calculator[self.var_type](
                    expected_delays,
                    src_lens,
                    target_padding_mask,
                    batch_first=True,
                    start_from_zero=False,
                )
            else:
                raise RuntimeError(f"{self.var_type} is not supported.")

            return self.var_weight * variance_delays.sum()
        else:
            return 0.0

    def loss(self, attention, source_padding_mask=None, target_padding_mask=None):
        expected_delays, src_lens = self.expected_delays_from_attention(
            attention, source_padding_mask, target_padding_mask
        )

        latency_loss = 0

        latency_loss += self.avg_loss(expected_delays, src_lens, target_padding_mask)

        latency_loss += self.var_loss(expected_delays, src_lens, target_padding_mask)

        return latency_loss
