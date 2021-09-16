# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from typing import List, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import lengths_to_mask
from fairseq.models.fairseq_model import FairseqEncoderModel


@dataclass
class FastSpeech2CriterionConfig(FairseqDataclass):
    ctc_weight: float = field(
        default=0.0, metadata={"help": "weight for CTC loss"}
    )


@register_criterion("fastspeech2", dataclass=FastSpeech2CriterionConfig)
class FastSpeech2Loss(FairseqCriterion):
    def __init__(self, task, ctc_weight):
        super().__init__(task)
        self.ctc_weight = ctc_weight

    def forward(self, model: FairseqEncoderModel, sample, reduction="mean"):
        src_tokens = sample["net_input"]["src_tokens"]
        src_lens = sample["net_input"]["src_lengths"]
        tgt_lens = sample["target_lengths"]
        _feat_out, _, log_dur_out, pitch_out, energy_out = model(
            src_tokens=src_tokens,
            src_lengths=src_lens,
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            incremental_state=None,
            target_lengths=tgt_lens,
            speaker=sample["speaker"],
            durations=sample["durations"],
            pitches=sample["pitches"],
            energies=sample["energies"]
        )

        src_mask = lengths_to_mask(sample["net_input"]["src_lengths"])
        tgt_mask = lengths_to_mask(sample["target_lengths"])

        pitches, energies = sample["pitches"], sample["energies"]
        pitch_out, pitches = pitch_out[src_mask], pitches[src_mask]
        energy_out, energies = energy_out[src_mask], energies[src_mask]

        feat_out, feat = _feat_out[tgt_mask], sample["target"][tgt_mask]
        l1_loss = F.l1_loss(feat_out, feat, reduction=reduction)

        pitch_loss = F.mse_loss(pitch_out, pitches, reduction=reduction)
        energy_loss = F.mse_loss(energy_out, energies, reduction=reduction)

        log_dur_out = log_dur_out[src_mask]
        dur = sample["durations"].float()
        dur = dur.half() if log_dur_out.type().endswith(".HalfTensor") else dur
        log_dur = torch.log(dur + 1)[src_mask]
        dur_loss = F.mse_loss(log_dur_out, log_dur, reduction=reduction)

        ctc_loss = torch.tensor(0.).type_as(l1_loss)
        if self.ctc_weight > 0.:
            lprobs = model.get_normalized_probs((_feat_out,), log_probs=True)
            lprobs = lprobs.transpose(0, 1)  # T x B x C
            src_mask = lengths_to_mask(src_lens)
            src_tokens_flat = src_tokens.masked_select(src_mask)
            ctc_loss = F.ctc_loss(
                lprobs, src_tokens_flat, tgt_lens, src_lens,
                reduction=reduction, zero_infinity=True
            ) * self.ctc_weight

        loss = l1_loss + dur_loss + pitch_loss + energy_loss + ctc_loss

        sample_size = sample["nsentences"]
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "l1_loss": utils.item(l1_loss.data),
            "dur_loss": utils.item(dur_loss.data),
            "pitch_loss": utils.item(pitch_loss.data),
            "energy_loss": utils.item(energy_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
        }
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        ns = [log.get("sample_size", 0) for log in logging_outputs]
        ntot = sum(ns)
        ws = [n / (ntot + 1e-8) for n in ns]
        for key in [
            "loss", "l1_loss", "dur_loss", "pitch_loss", "energy_loss",
            "ctc_loss"
        ]:
            vals = [log.get(key, 0) for log in logging_outputs]
            val = sum(val * w for val, w in zip(vals, ws))
            metrics.log_scalar(key, val, ntot, round=3)
        metrics.log_scalar("sample_size", ntot, len(logging_outputs))

        # inference metrics
        if "targ_frames" not in logging_outputs[0]:
            return
        n = sum(log.get("targ_frames", 0) for log in logging_outputs)
        for key, new_key in [
                ("mcd_loss", "mcd_loss"),
                ("pred_frames", "pred_ratio"),
                ("nins", "ins_rate"),
                ("ndel", "del_rate"),
        ]:
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(new_key, val / n, n, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
