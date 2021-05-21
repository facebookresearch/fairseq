# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from dataclasses import dataclass, field
import logging
import math
import os
from typing import Optional
import torch

from fairseq.logging import metrics
from fairseq.tasks import FairseqTask, register_task
from ..data import ExtractedFeaturesDataset, RandomInputDataset

from fairseq.data import (
    Dictionary,
    data_utils,
    StripTokenDataset,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.distributed.utils import get_data_parallel_world_size
from omegaconf import MISSING

from examples.speech_recognition.kaldi.kaldi_decoder import (
    KaldiDecoder,
    KaldiDecoderConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class DecodingConfig(FairseqDataclass):
    kenlm_path: Optional[str] = None
    lm_weight: float = 0
    blank_weight: float = 0


@dataclass
class UnpairedAudioTextConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory containing audio"}
    )
    text_data: str = field(
        default=MISSING, metadata={"help": "path to data directory containing text"}
    )
    max_length: Optional[int] = None
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    unfiltered: bool = field(
        default=False, metadata={"help": "load data with _unfiltered suffix"}
    )
    ctc_eval: bool = field(
        default=False, metadata={"help": "eval UER as if computed by CTC"}
    )
    sort_by_length: bool = field(
        default=True, metadata={"help": "sort examples by length of audio timesteps"}
    )
    shuffle: bool = field(default=True, metadata={"help": "shuffle examples"})
    append_eos: bool = field(default=False, metadata={"help": "append eos"})
    uppercase: Optional[bool] = field(
        default=False, metadata={"help": "uppercase for LM score computation"}
    )
    skipwords: Optional[str] = field(
        default="",
        metadata={
            "help": "comma-separated words to be removed for LM score computation"
        },
    )
    kenlm_path: Optional[str] = None
    vocab_usage_power: float = 2

    word_decoder_config: Optional[KaldiDecoderConfig] = None
    word_kenlm_path: Optional[str] = None

    decoding_config: DecodingConfig = DecodingConfig()


@register_task("gan_audio_pretraining_feats", dataclass=UnpairedAudioTextConfig)
class UnpairedAudioText(FairseqTask):
    """ """

    cfg: UnpairedAudioTextConfig

    def __init__(
        self,
        cfg: UnpairedAudioTextConfig,
        source_dictionary=None,
        target_dictionary=None,
    ):
        super().__init__(cfg)

        self._target_dictionary = target_dictionary
        self._source_dictionary = source_dictionary
        self.num_symbols = (
            len([s for s in target_dictionary.symbols if not s.startswith("madeup")])
            - target_dictionary.nspecial
        )
        self.sil_id = (
            target_dictionary.index("<SIL>") if "<SIL>" in target_dictionary else -1
        )
        self.kenlm = None
        if cfg.kenlm_path is not None:
            import kenlm

            self.kenlm = kenlm.Model(cfg.kenlm_path)

        self.word_kenlm = None
        if cfg.word_kenlm_path is not None:
            import kenlm

            self.word_kenlm = kenlm.Model(cfg.word_kenlm_path)

        self.uppercase = cfg.uppercase
        self.skipwords = set(cfg.skipwords.split(","))

        def str_postprocess(s):
            s = " ".join(w for w in s.split() if w not in self.skipwords)
            s = s.upper() if self.uppercase else s
            return s

        self.str_postprocess = str_postprocess
        self.compute_lm_score = lambda s: self.kenlm.score(self.str_postprocess(s))

        self.compute_word_score = None
        if cfg.word_decoder_config is not None:
            self.kaldi_decoder = KaldiDecoder(cfg.word_decoder_config, beam=10)

            def compute_word_score(logits, padding):
                res = self.kaldi_decoder.decode(logits, padding)
                for r in res:
                    r = r.result()
                    assert len(r) == 1
                    r = r[0]
                    yield r["score"], r["words"]

            self.compute_word_score = compute_word_score

    @classmethod
    def setup_task(cls, cfg: UnpairedAudioTextConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        dict_path = os.path.join(cfg.text_data, "dict.txt")
        if os.path.exists(dict_path):
            target_dictionary = Dictionary.load(dict_path)
        else:
            dict_path = os.path.join(cfg.data, f"dict.{cfg.labels}.txt")
            target_dictionary = Dictionary.load(dict_path)

        return cls(cfg, target_dictionary=target_dictionary)

    def optimizer_step(self, optimizer, model, update_num):
        if hasattr(model, "get_groups_for_update"):
            groups = model.get_groups_for_update(update_num)
            optimizer.step(groups={groups})
        else:
            optimizer.step()

    def valid_step(self, sample, model, criterion):
        res = model(
            **sample["net_input"],
            dense_x_only=True,
        )

        dense_x = res["logits"]
        padding_mask = res["padding_mask"]

        word_scores = None
        if self.compute_word_score is not None:
            word_scores = self.compute_word_score(dense_x.cpu(), padding_mask.cpu())

        z = dense_x.argmax(-1)
        z[padding_mask] = self.target_dictionary.pad()

        vocab_seen = torch.zeros(self.num_symbols, dtype=torch.bool)

        import editdistance

        c_err = 0
        c_len = 0
        pred_c_len = 0
        lm_score_sum = 0
        for i, (x, t, id) in enumerate(
            zip(
                z,
                sample["target"],
                sample["id"],
            )
        ):

            t = t[(t >= self.target_dictionary.nspecial)]
            x = x[
                (x >= self.target_dictionary.nspecial)
                & (x < (self.num_symbols + self.target_dictionary.nspecial))
            ]
            if self.sil_id >= 0:
                x = x[x != self.sil_id]

            vocab_seen[x - self.target_dictionary.nspecial] = True

            pred_units_arr = x
            if self.cfg.ctc_eval:
                pred_units_arr = pred_units_arr.unique_consecutive()
                pred_units_arr = pred_units_arr[pred_units_arr != 0]

            if id == 0:
                logger.info(f"REF: {self.target_dictionary.string(t)}")
                logger.info(f"HYP: {self.target_dictionary.string(pred_units_arr)}")

                if self.kenlm is not None:
                    ref_lm_s = self.compute_lm_score(self.target_dictionary.string(t))
                    hyp_lm_s = self.compute_lm_score(
                        self.target_dictionary.string(pred_units_arr)
                    )
                    logger.info(
                        f"LM [REF]: {ref_lm_s}, {math.pow(10, ref_lm_s / (len(t) + 1))}"
                    )
                    logger.info(
                        f"LM [HYP]: {hyp_lm_s}, {math.pow(10, hyp_lm_s / (len(pred_units_arr) + 1))}"
                    )

            pred_units_arr = pred_units_arr.tolist()

            t = t.tolist()
            c_err += editdistance.eval(pred_units_arr, t)
            c_len += len(t)
            pred_c_len += len(pred_units_arr)

            if self.kenlm is not None:
                pred_str = self.target_dictionary.string(pred_units_arr)
                lm_score = self.compute_lm_score(pred_str)
                lm_score_sum += lm_score

        kaldi_score_sum = 0
        word_lm_sum = 0
        num_words = 0
        if word_scores is not None:
            for score, words in word_scores:
                kaldi_score_sum += score
                num_words += len(words)
                if self.word_kenlm is not None:
                    word_lm_sum += self.kenlm.score(" ".join(words))

        try:
            world_size = get_data_parallel_world_size()
        except:
            world_size = 1

        logging_output = {
            "loss": c_err,
            "_num_char_errors": c_err,
            "_num_chars": c_len,
            "_num_pred_chars": pred_c_len,
            "ntokens": c_len,
            "nsentences": z.size(0),
            "sample_size": c_len,
            "_world_size": world_size,
            "_lm_score_sum": lm_score_sum,
            "_kaldi_score_sum": kaldi_score_sum,
            "_word_lm_sum": word_lm_sum,
            "_num_words": num_words,
            "_vocab_seen": vocab_seen,
        }

        return c_err, c_len, logging_output

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        has_unpaired_text = os.path.exists(
            os.path.join(self.cfg.text_data, f"{split}.idx")
        )

        self.datasets[split] = ExtractedFeaturesDataset(
            path=data_path,
            split=split,
            min_length=3,
            max_length=task_cfg.max_length,
            labels=None if has_unpaired_text else task_cfg.labels,
            label_dict=self.target_dictionary,
            shuffle=getattr(task_cfg, "shuffle", True),
            sort_by_length=task_cfg.sort_by_length,
        )

        logger.info(f"split {split} has unpaired text? {has_unpaired_text}")
        if has_unpaired_text:
            text_dataset = data_utils.load_indexed_dataset(
                os.path.join(self.cfg.text_data, split), self.target_dictionary
            )
            text_dataset = StripTokenDataset(text_dataset, self.target_dictionary.eos())
            self.datasets[split] = RandomInputDataset(
                self.datasets[split],
                text_dataset,
                ["random_label"],
                add_to_input=True,
                pad_idx=self.target_dictionary.pad(),
            )

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        num_pred_chars = sum(
            log.get("_num_pred_chars", zero) for log in logging_outputs
        )

        lm_score_sum = sum(log.get("_lm_score_sum", zero) for log in logging_outputs)
        vocab_seen = (
            sum(log.get("_vocab_seen", zero) for log in logging_outputs)
            .bool()
            .sum()
            .item()
        )
        kaldi_score_sum = sum(
            log.get("_kaldi_score_sum", zero) for log in logging_outputs
        )
        word_lm_sum = sum(log.get("_word_lm_sum", zero) for log in logging_outputs)

        metrics.log_scalar_sum("_num_char_errors", num_char_errors)
        metrics.log_scalar_sum("_num_chars", num_chars)
        metrics.log_scalar_sum("_num_word_errors", num_word_errors)
        metrics.log_scalar_sum("_num_words", num_words)

        metrics.log_scalar_sum("lm_score_sum", lm_score_sum)
        metrics.log_scalar_sum("num_pred_chars", num_pred_chars)

        if self.cfg.word_kenlm_path is not None:
            metrics.log_scalar_sum("kaldi_score_sum", kaldi_score_sum)
            metrics.log_scalar_sum("word_lm_sum", word_lm_sum)

        if num_chars > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )

            if lm_score_sum < 0 and vocab_seen > 0:
                metrics.log_scalar("vocab_seen_pct", vocab_seen / self.num_symbols)

                metrics.log_derived(
                    "weighted_lm_ppl",
                    lambda meters: math.pow(
                        10,
                        -meters["lm_score_sum"].sum
                        / (
                            meters["num_pred_chars"].sum + meters["nsentences"].sum
                        ),  # account for </s>
                    )
                    / meters["vocab_seen_pct"].avg ** self.cfg.vocab_usage_power,
                )

                metrics.log_derived(
                    "lm_ppl",
                    lambda meters: math.pow(
                        10,
                        -meters["lm_score_sum"].sum
                        / (
                            meters["num_pred_chars"].sum + meters["nsentences"].sum
                        ),  # account for </s>
                    ),
                )
            else:
                metrics.log_derived("weighted_lm_ppl", lambda meters: float("inf"))

        if num_words > 0:
            if word_lm_sum != 0:
                metrics.log_derived(
                    "word_lm_ppl",
                    lambda meters: math.pow(
                        10,
                        -meters["word_lm_sum"].sum
                        / (
                            meters["_num_words"].sum + meters["nsentences"].sum
                        ),  # account for </s>
                    ),
                )
                metrics.log_derived(
                    "weighted_word_lm_ppl",
                    lambda meters: math.pow(
                        10,
                        -meters["word_lm_sum"].sum
                        / (
                            meters["_num_words"].sum + meters["nsentences"].sum
                        ),  # account for </s>
                    )
                    / meters["vocab_seen_pct"].avg ** self.cfg.vocab_usage_power,
                )

            if self.cfg.word_kenlm_path is not None:
                metrics.log_derived(
                    "kaldi_score",
                    lambda meters: meters["kaldi_score_sum"].sum
                    / meters["nsentences"].sum,
                )

    def build_model(self, cfg: FairseqDataclass):
        model = super().build_model(cfg)

        return model
