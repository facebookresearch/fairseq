#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.logging.meters import safe_round

from .multi_modality_cross_entropy import SpeechTextPreTrainCrossEntCriterion

logger = logging.getLogger(__name__)


@dataclass
class SpeechTextPreTrainCompoundCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    post_process: str = field(
        default="none",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )


@register_criterion(
    "speech_text_pretrain_compound", dataclass=SpeechTextPreTrainCompoundCriterionConfig
)
class SpeechTextPreTrainCompoundCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        report_accuracy=False,
        zero_infinity=False,
        post_process=None,
    ):
        super().__init__(task)
        self.xent = SpeechTextPreTrainCrossEntCriterion(
            task, sentence_avg, label_smoothing, report_accuracy
        )
        cfg_dict = {
            "zero_infinity": zero_infinity,
            "sentence_avg": sentence_avg,
            "post_process": post_process,
        }
        cfg_ctc = CtcCriterionConfig(**cfg_dict)
        self.ctc = CtcCriterion(cfg_ctc, task)

    def forward(self, model, sample, reduce=True):
        mode = sample["net_input"]["mode"]
        if mode == "sup_speech_ctc":  # CTC
            sample["net_input"][
                "src_lengths"
            ] = None  # get downsampled src_lengths from padding_mask
            loss, sample_size, logging_output = self.ctc(model, sample, reduce)
            logging_output["mode"] = SpeechTextPreTrainCompoundCriterion.mode2value(
                "CTC"
            )
        else:
            loss, sample_size, logging_output = self.xent(model, sample, reduce)
            logging_output["mode"] = SpeechTextPreTrainCompoundCriterion.mode2value(
                "xent"
            )

        return loss, sample_size, logging_output

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    @staticmethod
    def mode2value(mode):  # make the logging_outputs_can_be_summed = True
        if mode == "CTC":
            return 907  # prime number
        if mode == "xent":
            return 887  # prime number
        return 0

    @staticmethod
    def value2mode(value):
        if value % 907 == 0:
            return "CTC"
        if value % 887 == 0:
            return "xent"
        raise ValueError("Unknow mode")

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        def _get_mode(logging_outputs):
            mds = [
                SpeechTextPreTrainCompoundCriterion.value2mode(log["mode"])
                for log in logging_outputs
            ]
            if sum([1 if l != mds[0] else 0 for l in mds]) > 0:
                raise ValueError("mode in one mini-batch is expected to be the same!")
            return mds[0]

        log_mode = _get_mode(logging_outputs)
        if log_mode == "xent":
            return SpeechTextPreTrainCrossEntCriterion.reduce_metrics(logging_outputs)

        # ctc loss
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "ctc_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ctc_ntokens", ntokens)
        metrics.log_scalar("ctc_nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "ctc_nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
