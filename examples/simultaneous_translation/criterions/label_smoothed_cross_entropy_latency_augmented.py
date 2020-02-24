# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import utils

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion
)

from examples.simultaneous_translation.utils.latency import (
    LatencyTraining
)

@register_criterion('latency_augmented_label_smoothed_cross_entropy')
class LatencyAugmentedLabelSmoothedCrossEntropyCriterion(
    LabelSmoothedCrossEntropyCriterion
):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.latency_weight_avg = args.latency_weight_avg
        self.latency_weight_avg_type = args.latency_weight_avg_type
        self.latency_weight_var = args.latency_weight_var
        self.latency_weight_var_type = args.latency_weight_var_type
        self.latency_weight_non_trivial = args.latency_weight_non_trivial
        self.mass_preservation = args.mass_preservation
        self.average_method = args.average_method
        self.latency_train = LatencyTraining(
            self.latency_weight_avg,
            self.latency_weight_var,
            self.latency_weight_avg_type,
            self.latency_weight_var_type,
            self.mass_preservation,
            self.average_method,
            args.var_power,
            args.var_span
        )

    @staticmethod
    def add_args(parser):
        super(
            LatencyAugmentedLabelSmoothedCrossEntropyCriterion,
            LatencyAugmentedLabelSmoothedCrossEntropyCriterion
        ).add_args(parser)
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--latency-weight-avg", default=0., type=float, metavar='D',
                            help="")
        parser.add_argument("--latency-weight-var", default=0., type=float, metavar='D',
                            help="")
        parser.add_argument("--latency-weight-avg-type", default="differentiable_average_lagging",
                            help="")
        parser.add_argument("--latency-weight-var-type", default="variance_delay",
                            help="")
        parser.add_argument("--latency-weight-non-trivial", default=0., type=float, metavar='D',
                            help="")
        parser.add_argument("--average-method", default="weighted_average",
                            help="")
        parser.add_argument("--var-power", type=float, default=2.0,
                            help="")
        parser.add_argument("--var-span", type=float, default=1.0,
                            help="")
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, latency_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'latency_loss': latency_loss
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        loss, nll_loss = super().compute_loss(model, net_output, sample, reduce)
        if type(net_output[-1]) is dict:
            attn_list = [item["alpha"] for item in net_output[-1]["attn_list"]]

            target_padding_mask = (
                model
                .get_targets(sample, net_output)
                .eq(self.padding_idx)
            )
            source_padding_mask = (
                net_output[-1].get("encoder_padding_mask", None)
            )

            latency_loss = self.latency_train.loss(
                attn_list,
                source_padding_mask,
                target_padding_mask,
            )
            loss += latency_loss

        else:
            latency_term = net_output[-1]
            latency_loss = self.latency_weight_avg * latency_term.sum()
            loss += latency_loss

        return loss, nll_loss, latency_loss
