# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import editdistance
import os
import sys
import torch

from fairseq.data import AddTargetDataset, Dictionary, FileAudioDataset, encoders
from fairseq.data.data_utils import post_process

from . import LegacyFairseqTask, register_task
from .. import utils
from ..logging import metrics


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@register_task("audio_pretraining")
class AudioPretrainingTask(LegacyFairseqTask):
    """"""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=int,
            help="target sample rate. audio files will be up/down sampled to this rate",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--max-sample-size",
            default=None,
            type=int,
            help="max sample size to crop to for batching. default = min sample length",
        )
        parser.add_argument(
            "--min-sample-size",
            default=None,
            type=int,
            help="min sample size to crop to for batching. default = same as --max-sample-size",
        )

        parser.add_argument(
            "--enable-padding",
            action="store_true",
            help="pad shorter samples instead of cropping",
        )

        parser.add_argument(
            "--labels",
            type=str,
            default=None,
            help="extension of the label file to load, if any",
        )

        # Options for reporting WER metrics during validation. Only applicable to
        # Seq2Seq models during fine-tuning
        parser.add_argument(
            "--eval-wer",
            action="store_true",
            help="compute WER for Seq2Seq models",
        )
        parser.add_argument(
            "--eval-wer-remove-bpe",
            default="letter",
            help="remove BPE tokens before scoring (can be sentencepiece, letter, and more)",
        )

    def __init__(self, args, source_dictionary=None, target_dictionary=None):
        super().__init__(args)
        self._target_dictionary = target_dictionary
        self._source_dictionary = source_dictionary
        self.is_ctc = args.criterion == "ctc"
        if getattr(self.args, "eval_wer", False):
            assert args.labels is not None, "eval_wer can only be set during fine-tuning"

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (omegaconf.DictConfig): parsed command-line arguments
        """

        if args.labels:
            dict_path = os.path.join(args.data, f"dict.{args.labels}.txt")
            target_dictionary = Dictionary.load(dict_path)
        else:
            target_dictionary = None

        return cls(args, target_dictionary=target_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.max_sample_size,
            min_length=self.args.min_sample_size,
            pad=self.args.labels is not None or self.args.enable_padding,
            normalize=self.args.normalize,
        )

        if self.args.labels:
            label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    labels.append(line)

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=not self.is_ctc,
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
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if getattr(self.args, "eval_wer", False) and not self.is_ctc:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, args):
        model = super().build_model(args)

        if getattr(args, 'eval_wer', False) and not self.is_ctc:
            self.sequence_generator = self.build_generator([model], args, )
            self.tokenizer = encoders.build_tokenizer(args)
        return model

    def _inference_with_wer(self, generator, sample, model):
        def decode(toks, escape_unk=True):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.args.eval_wer_remove_bpe,
                escape_unk=escape_unk,
                extra_symbols_to_ignore={generator.eos},
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
                escape_unk=True,
            )
            hyp = post_process(hyp, self.args.eval_wer_remove_bpe).strip("_")
            ref = post_process(ref, self.args.eval_wer_remove_bpe).strip("_")
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split("_")
            ref_words = ref.split("_")
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)
        
        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.)
        num_char_errors = sum(log.get("_num_char_errors", zero) for log in logging_outputs)
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(log.get("_num_word_errors", zero) for log in logging_outputs)
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_words > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum * 100.0 / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0 else float("nan")
            )
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum * 100.0 / meters["_num_words"].sum
                if meters["_num_words"].sum > 0 else float("nan")
            )
