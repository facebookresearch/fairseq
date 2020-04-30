# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        self.mass_preservation = args.mass_preservation
        self.average_method = args.average_method
        self.latency_train = LatencyTraining(
            self.latency_weight_avg,
            self.latency_weight_var,
            self.latency_weight_avg_type,
            self.latency_weight_var_type,
            self.mass_preservation,
            self.average_method,
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
                            help="Average loss weight")
        parser.add_argument("--latency-weight-var", default=0., type=float, metavar='D',
                            help="Variance loss weight")
        parser.add_argument("--latency-weight-avg-type", default="differentiable_average_lagging",
                            help="Statistics for Average loss type")
        parser.add_argument("--latency-weight-var-type", default="variance_delay",
                            help="Statistics for variance loss type")
        parser.add_argument("--average-method", default="weighted_average",
                            help="Average loss type")
        # fmt: on

    def compute_loss(self, model, net_output, sample, reduce=True):
        # Compute cross entropy loss first
        loss, nll_loss = super().compute_loss(model, net_output, sample, reduce)

        # Obtain the expected alignment
        attn_list = [item["alpha"] for item in net_output[-1]["attn_list"]]

        target_padding_mask = model.get_targets(sample, net_output).eq(self.padding_idx)

        source_padding_mask = net_output[-1].get("encoder_padding_mask", None)

        # Get latency loss
        latency_loss = self.latency_train.loss(
            attn_list, source_padding_mask, target_padding_mask)

        loss += latency_loss

        return loss, nll_loss
