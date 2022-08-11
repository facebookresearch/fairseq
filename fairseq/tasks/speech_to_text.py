# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from argparse import Namespace

import numpy as np

from fairseq import utils, metrics, scoring
from fairseq.utils import safe_hasattr
from fairseq.data import Dictionary, encoders
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
)
from fairseq.tasks import LegacyFairseqTask, register_task

EVAL_BLEU_ORDER=4

logger = logging.getLogger(__name__)


@register_task("speech_to_text")
class SpeechToTextTask(LegacyFairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--eval-print-samples",
            default=False,
            action="store_true",
            help="print sample generations during validation",
        )

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()
        self.pre_tokenizer = self.build_tokenizer(args)
        self.bpe_tokenizer = self.build_bpe(args)
        self.scorer = scoring.build_scorer(args, self.tgt_dict)
        if (
            self.data_cfg.prepend_tgt_lang_tag
            and self.data_cfg.prepend_bos_and_append_tgt_lang_tag
        ):
            raise ValueError(
                "Please set only one of the two options to avoid adding target token multiple times"
            )

    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.args.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        dict_path = Path(args.data) / data_cfg.vocab_filename
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        if args.scoring == "wer":
            if args.wer_tokenizer == "none":
                logger.warning(
                    "You are not using any tokenizer for WER scoring. Using '13a' is recommended."
                )
            if not args.wer_lowercase or not args.wer_remove_punct:
                logger.warning(
                    "You are not lowercasing or removing punctuation before WER scoring."
                )

        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            self.pre_tokenizer,
            self.bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=self.speaker_to_id,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args, from_checkpoint=False):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        args.speaker_to_id = self.speaker_to_id
        model = super(SpeechToTextTask, self).build_model(args, from_checkpoint)
        self.sequence_generator = self.build_generator([model], args)
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        def decode(toks):
            if hasattr(self.sequence_generator, "symbols_to_strip_from_output"):
                to_ignore = self.sequence_generator.symbols_to_strip_from_output
            else:
                to_ignore = {self.sequence_generator.eos}

            s = self.tgt_dict.string(
                toks.int().cpu(),
                escape_unk=True,
                extra_symbols_to_ignore=to_ignore
            )
            if self.bpe_tokenizer:
                s = self.bpe_tokenizer.decode(s)
            if self.pre_tokenizer:
                s = self.pre_tokenizer.decode(s)
            return s

        if self.scorer:
            prefix_tokens = sample['target'][:, 0].unsqueeze(1) if self.data_cfg.prepend_tgt_lang_tag else None
            gen_out = self.inference_step(self.sequence_generator, [model], sample, prefix_tokens=prefix_tokens)
            for i in range(len(gen_out)):
                ref_tok = utils.strip_pad(sample["target"][i], self.tgt_dict.pad()).int().cpu()
                pred_tok = gen_out[i][0]["tokens"].int().cpu()
                if self.data_cfg.prepend_tgt_lang_tag:
                    ref_tok = ref_tok[1:]
                    pred_tok = pred_tok[1:]
                ref = decode(ref_tok)
                pred = decode(pred_tok)
                self.scorer.add_string(ref, pred)

            if self.args.eval_print_samples:
                logger.info("Validation example:")
                logger.info("H-{} {}".format(sample["id"][-1], pred))
                logger.info("T-{} {}".format(sample["id"][-1], ref))

        if self.args.scoring == "wer":
            logging_output["_wer_distance"] = self.scorer.distance
            logging_output["_wer_ref_len"] = self.scorer.ref_length
        elif self.args.scoring == "sacrebleu":
            sacrebleu_out = self.scorer._score()
            logging_output["_bleu_sys_len"] = sacrebleu_out.sys_len
            logging_output["_bleu_ref_len"] = sacrebleu_out.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(sacrebleu_out.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = sacrebleu_out.counts[i]
                logging_output["_bleu_totals_" + str(i)] = sacrebleu_out.totals[i]
        else:
            raise NotImplemented()

        if safe_hasattr(self.scorer, "reset"):
            self.scorer.reset()
        else:
            self.scorer.ref = []
            self.scorer.pred = []

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        if self.args.scoring == "wer":
            if  sum_logs("_wer_ref_len") > 0:
                metrics.log_scalar("_wer_distance", sum_logs("_wer_distance"))
                metrics.log_scalar("_wer_ref_len", sum_logs("_wer_ref_len"))

                def compute_wer(meters):
                    import torch
                    ref_len = meters["_wer_ref_len"].sum
                    wer = meters["_wer_distance"].sum / ref_len
                    if torch.is_tensor(wer):
                        wer = wer.cpu().item()
                    return round(100 * wer, 2)

                metrics.log_derived("wer", compute_wer)

        elif self.args.scoring == "sacrebleu":
            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import torch

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum if torch.is_tensor(meters["_bleu_sys_len"].sum) == False else meters["_bleu_sys_len"].sum.long().item(),
                        ref_len=meters["_bleu_ref_len"].sum if torch.is_tensor(meters["_bleu_ref_len"].sum) == False else meters["_bleu_ref_len"].sum.long().item(),
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("sacrebleu", compute_bleu)

        else:
            raise NotImplemented()

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}
        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids

        eos_token = (
            args.eos_token
            if "eos_token" in args and args.eos_token is not None
            else self.data_cfg.config.get("eos_token", None)
        )

        if self.data_cfg.prepend_bos_and_append_tgt_lang_tag and not eos_token:
            raise Warning(
                "Please provide --eos_token to replace eos in sequence generator"
            )

        eos_id = self.tgt_dict.index(eos_token) if eos_token else None
        extra_gen_cls_kwargs["eos"] = eos_id

        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
