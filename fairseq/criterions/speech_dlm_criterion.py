# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class SpeechDLMCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    main_and_cross_weights: Optional[str] = field(
        default="1,0",
        metadata={
            "help": "Comma-separated list of weights of Main-channel vs Cross-channel Prediction Losses"
            "(default: 1,0)"
        },
    )
    general_unit_loss_weight: float = field(
        default=0,
        metadata={
            "help": "The weight of the General Prediction Loss (Next-step Unit Prediction Loss)"
            "(default: 0)"
        },
    )
    edge_unit_loss_weight: float = field(
        default=1,
        metadata={"help": "The weight of the Edge Unit Prediction Loss" "(default: 1)"},
    )
    duration_loss_weight: float = field(
        default=1,
        metadata={
            "help": "The weight of the Edge Unit Duration Prediction Loss"
            "(default: 1)"
        },
    )


@register_criterion("speech_dlm_criterion", dataclass=SpeechDLMCriterionConfig)
class SpeechDLMCriterion(FairseqCriterion):
    """Criteron for the SpeechDLM model as described in the paper:
    https://arxiv.org/pdf/2203.16502.pdf

    There are 3 possible losses depending on the targets of the model:
        - general_unit_loss : The next unit prediction loss, corresponding to
            'next' target
        - edge_unit_loss : The edge unit prediction loss, corresponding to
            'edge' target
        - duration_loss : The duration prediction loss, corresponding to
            'duration' target
    """

    def __init__(
        self,
        task,
        sentence_avg,
        main_and_cross_weights,
        general_unit_loss_weight,
        edge_unit_loss_weight,
        duration_loss_weight,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg

        self.channels = task.channels
        self.targets = task.targets
        self.delayed_duration_target = task.delayed_duration_target

        self.main_channel_weight = float(main_and_cross_weights.split(",")[0])
        self.cross_channel_weight = float(main_and_cross_weights.split(",")[1])
        assert self.main_channel_weight >= 0 and self.cross_channel_weight >= 0

        self.channel_weights = {
            channel: weight
            for channel, weight in zip(self.channels, task.channel_weights)
        }

        self.target_weights = {}
        for t in self.targets:
            if t == "next":
                self.target_weights[t] = general_unit_loss_weight
                assert (
                    general_unit_loss_weight > 0
                ), "Expect a positive --general-unit-loss-weight for next unit prediction"
            elif t == "edge":
                self.target_weights[t] = edge_unit_loss_weight
                assert (
                    edge_unit_loss_weight > 0
                ), "Expect a positive --edge-unit-loss-weight for edge unit prediction"
            elif t == "duration":
                self.target_weights[t] = duration_loss_weight
                assert (
                    duration_loss_weight > 0
                ), "Expect a positive --duration-loss-weight for duration prediction"

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss_dict, stats_dict = self.compute_loss(
            model, net_output, sample, reduce=reduce
        )
        nsentences = sample["net_input"]["src_tokens"][self.channels[0]].size(0)

        logging_output = {
            "nsentences": nsentences,
        }
        logging_output["nsentences"] = nsentences

        loss_all = {t: 0 for t in self.targets}
        correct_all = {t: 0 for t in self.targets}
        count_all = {t: 0 for t in self.targets}
        ntokens_all = 0
        sample_size_all = 0
        for channel in loss_dict:
            for pred_channel in loss_dict[channel]:
                # Get ntokens & sample_size
                ntokens = sample["net_input"]["src_tokens"][channel].numel()
                sample_size = nsentences if self.sentence_avg else ntokens
                prefix = "[{}-{}]".format(channel, pred_channel)
                log_keys = {
                    "next": "general_token",
                    "edge": "edge_token",
                    "duration": "edge_duration",
                }

                # Log & Update the sizes
                logging_output["{}ntokens".format(prefix)] = ntokens
                logging_output["{}sample_size".format(prefix)] = sample_size
                ntokens_all += ntokens
                sample_size_all += sample_size

                for t in self.targets:
                    log_key = log_keys[t]
                    loss = loss_dict[channel][pred_channel][t]
                    correct, count = stats_dict[channel][pred_channel][t]

                    # Log the statistics
                    logging_output["{}{}_loss".format(prefix, log_key)] = loss.data
                    logging_output["{}{}_correct".format(prefix, log_key)] = correct
                    logging_output["{}{}_count".format(prefix, log_key)] = count

                    # Scale the training loss by weights
                    target_loss = loss * self.channel_weights[channel]
                    if pred_channel == channel:
                        target_loss = target_loss * self.main_channel_weight
                    else:
                        target_loss = target_loss * self.cross_channel_weight
                    # Normalize the losses in the training by the number of edges
                    if t in ["edge", "duration"]:
                        target_loss = target_loss / count * sample_size

                    # Update the statistics
                    loss_all[t] += target_loss
                    correct_all[t] += correct
                    count_all[t] += count

        # Logging the average statistics
        logging_output["ntokens"] = ntokens_all
        logging_output["sample_size"] = sample_size_all
        for t in self.targets:
            log_key = {
                "next": "general_token",
                "edge": "edge_token",
                "duration": "edge_duration",
            }[t]
            logging_output["{}_loss".format(log_key)] = loss_all[t].data
            logging_output["{}_correct".format(log_key)] = correct_all[t]
            logging_output["{}_count".format(log_key)] = count_all[t]

        # Define the training loss
        training_loss = 0
        for t in self.targets:
            training_loss += loss_all[t] * self.target_weights[t]
        logging_output["loss"] = training_loss.data

        return training_loss, sample_size_all, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # Get the model outputs and target
        lprobs_dict = model.get_normalized_probs(net_output, log_probs=True)
        target_dict = model.get_targets(sample, net_output)

        # Init the dictionaries
        loss_dict, stats_dict = {}, {}

        for channel in lprobs_dict:
            # Init the dictionaries
            loss_dict[channel], stats_dict[channel] = {}, {}

            for pred_channel in lprobs_dict[channel]:
                # Init the dictionaries
                loss_dict[channel][pred_channel] = {}
                stats_dict[channel][pred_channel] = {}

                # Get token & duration predictions
                outputs = lprobs_dict[channel][pred_channel]
                if not isinstance(outputs, dict):
                    token_lprobs = outputs
                else:
                    token_lprobs = outputs["pred_token"]
                    dur_preds = outputs["pred_duration"]
                    dur_preds = dur_preds.view(-1)
                token_lprobs = token_lprobs.view(-1, token_lprobs.size(-1))
                token_preds = token_lprobs.argmax(dim=-1)

                # Get edge indices
                if "edge" in self.targets or "duration" in self.targets:
                    edge_indices = target_dict["edge_indices"][pred_channel]

                # Compute loss and statistics
                for t in self.targets:
                    if t in ["next", "edge"]:
                        if t == "next":
                            target = target_dict["next"][pred_channel].view(-1)
                            lprobs = token_lprobs
                            preds = token_preds
                        elif t == "edge":
                            target = target_dict["edge"][pred_channel]
                            lprobs = token_lprobs[edge_indices]
                            preds = token_preds[edge_indices]

                        loss = F.nll_loss(
                            lprobs,
                            target,
                            ignore_index=self.padding_idx,
                            reduction="sum" if reduce else "none",
                        )
                    elif t == "duration":
                        target = target_dict["duration"][pred_channel]
                        if self.delayed_duration_target:
                            duration_indices = edge_indices + 1
                            if duration_indices[-1] == len(dur_preds):
                                duration_indices = duration_indices[:-1]
                                target = target[:-1]
                        else:
                            duration_indices = edge_indices
                        preds = dur_preds[duration_indices]

                        loss = F.l1_loss(
                            preds,
                            target,
                            reduction="sum" if reduce else "none",
                        )
                        preds = preds.round()

                    correct = (preds == target).sum().float().cpu().item()
                    count = float(target.size(0))

                    loss_dict[channel][pred_channel][t] = loss
                    stats_dict[channel][pred_channel][t] = (correct, count)

        return loss_dict, stats_dict

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        logging_keys = next(iter(logging_outputs)).keys()
        channels = [item[:-7] for item in logging_keys if item.endswith("ntokens")]
        target_prefixes = set(
            [
                item[:-5].split("]")[-1]
                for item in logging_keys
                if item.endswith("_loss")
            ]
        )
        for channel_prefix in channels:
            for target_prefix in target_prefixes:
                prefix = "{}{}".format(channel_prefix, target_prefix)
                count_sum = sum(
                    log.get("{}_count".format(prefix), 0) for log in logging_outputs
                )
                correct_sum = sum(
                    log.get("{}_correct".format(prefix), 0) for log in logging_outputs
                )
                loss_sum = sum(
                    log.get("{}_loss".format(prefix), 0) for log in logging_outputs
                )

                if "duration" not in target_prefix:
                    # we divide by log(2) to convert the loss from base e to base 2
                    metrics.log_scalar(
                        "{}_loss".format(prefix),
                        loss_sum / count_sum / math.log(2),
                        count_sum,
                        round=3,
                    )
                    metrics.log_derived(
                        "{}_ppl".format(prefix),
                        lambda meters, prefix=prefix: utils.get_perplexity(
                            meters["{}_loss".format(prefix)].avg
                        ),
                    )
                else:
                    # for duration we don't need to divide by log(2)
                    metrics.log_scalar(
                        "{}_loss".format(prefix),
                        loss_sum / count_sum,
                        count_sum,
                        round=3,
                    )

                accuracy = 100 * correct_sum / count_sum
                metrics.log_scalar("{}_pred_acc".format(prefix), accuracy, round=3)

        # Logging training loss
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
