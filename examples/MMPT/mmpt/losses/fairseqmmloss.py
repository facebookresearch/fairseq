# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
TODO (huxu): a general fairseq criterion for all your pre-defined losses.
"""

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics


@register_criterion("mmloss")
class MMCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)
        # TODO (huxu): wrap forward call of loss_fn and eval_fn into task.
        self.mmtask = task.mmtask

    def forward(self, model, sample):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        outputs = self.mmtask(model, sample)

        loss, loss_scalar, max_len, batch_size, sample_size = (
            outputs["loss"],
            outputs["loss_scalar"],
            outputs["max_len"],
            outputs["batch_size"],
            outputs["sample_size"],
        )

        logging_output = {
            "loss": loss_scalar,
            "ntokens": max_len * batch_size,  # dummy report.
            "nsentences": batch_size,  # dummy report.
            "sample_size": sample_size,
        }

        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        """since we use NCE, our actual batch_size is 1 per GPU.
        Then we take the mean of each worker."""
        loss_sum = sum(log.get("loss", 0.0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
