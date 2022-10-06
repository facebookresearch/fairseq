# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import Namespace
from pathlib import Path
from typing import List

import torch

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.data_cfg import MultitaskConfig
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    TextTargetMultitaskData,
)
from fairseq.logging import metrics
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.utils import strip_pad

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
            "--multitask-config-yaml",
            type=str,
            default=None,
            help="Configuration YAML filename for the multitasks (under manifest root)",
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
        parser.add_argument("--eval-bleu", action="store_true", default=False)
        parser.add_argument(
            "--eval-bleu-print-samples", action="store_true", default=False
        )
        parser.add_argument("--eval-bleu-remove-bpe", type=str, default=None)
        parser.add_argument("--eval-bleu-tokenizer", type=str, default="13a")
        parser.add_argument("--eval-bleu-beam", type=int, default=4)
        parser.add_argument("--eval-bleu-max-len-a", type=int, default=0)
        parser.add_argument("--eval-bleu-max-len-b", type=int, default=200)
        parser.add_argument("--ds-drop-probability", type=float, default=0.0)

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.cfg = args
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()
        if (
            self.data_cfg.prepend_tgt_lang_tag
            and self.data_cfg.prepend_bos_and_append_tgt_lang_tag
        ):
            raise ValueError(
                "Please set only one of the two options to avoid adding target token multiple times"
            )

        self.multitask_tasks = {}
        self.tgt_dict_mt = None
        self.eos_token_mt = None
        if getattr(args, "multitask_config_yaml", None) is not None:
            multitask_cfg = MultitaskConfig(
                Path(args.data) / args.multitask_config_yaml
            )
            for task_name, task_config in multitask_cfg.get_all_tasks().items():
                task_obj = DummyMultiTask(task_config, task_config.tgt_dict)
                self.multitask_tasks[task_name] = task_obj
                if "target" in task_name and task_config.decoder_type != "ctc":
                    self.tgt_dict_mt = task_obj.target_dictionary
                    if task_config.prepend_bos_and_append_tgt_lang_tag:
                        self.eos_token_mt = task_config.eos_token
                        assert not isinstance(self.eos_token_mt, List)

                        if not self.eos_token_mt:
                            raise Warning(
                                "Please provide --eos_token to replace eos in sequence generator"
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
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=self.speaker_to_id,
            multitask=self.multitask_tasks,
            drop_probability=self.args.ds_drop_probability,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def target_dictionary_mt(self):
        return self.tgt_dict_mt

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
        self.generator = None
        if getattr(self.cfg, "eval_bleu", False):
            args.beam = getattr(args, "beam", self.cfg.eval_bleu_beam)
            args.max_len_a = getattr(args, "max_len_a", self.cfg.eval_bleu_max_len_a)
            args.max_len_b = getattr(args, "max_len_b", self.cfg.eval_bleu_max_len_b)
            self.generator = self.build_generator([model], args)
        return model

    def build_generator_translatotron2(
        self,
        models,
        args,
        extra_gen_cls_kwargs,
    ):
        from fairseq.sequence_generator_multi_decoder import (
            MultiDecoderSequenceGenerator,
        )

        lang_token_ids_aux = {
            i
            for s, i in self.tgt_dict_mt.indices.items()
            if TextTargetMultitaskData.is_lang_tag(s)
        }

        extra_gen_cls_kwargs["symbols_to_strip_from_output"].update(lang_token_ids_aux)

        eos_id_mt = (
            self.tgt_dict_mt.index(self.eos_token_mt) if self.eos_token_mt else None
        )
        assert eos_id_mt != self.tgt_dict_mt.unk()
        extra_gen_cls_kwargs["eos_mt"] = eos_id_mt

        return MultiDecoderSequenceGenerator(
            models,
            self.target_dictionary,
            self.target_dictionary_mt,
            beam_size=max(1, getattr(args, "beam", 1)),
            beam_size_mt=max(1, getattr(args, "beam_mt", 1)),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            max_len_a_mt=getattr(args, "max_len_a_mt", 0),
            max_len_b_mt=getattr(args, "max_len_b_mt", 0),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            len_penalty_mt=getattr(args, "lenpen_mt", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            **extra_gen_cls_kwargs,
        )

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

        if self.data_cfg.prepend_bos_and_append_tgt_lang_tag:
            eos_token = (
                args.eos_token
                if "eos_token" in args and args.eos_token is not None
                else self.data_cfg.config.get("eos_token", None)
            )
            if not eos_token:
                raise Warning(
                    "Please provide --eos_token to replace eos in sequence generator"
                )

            eos_id = self.tgt_dict.index(eos_token) if eos_token else None
            extra_gen_cls_kwargs["eos"] = eos_id

        from fairseq.models.speech_to_text import XMTransformerUnitYModel

        if isinstance(models[0], XMTransformerUnitYModel):
            return self.build_generator_translatotron2(
                models,
                args,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            )
        else:
            return super().build_generator(
                models,
                args,
                seq_gen_cls=None,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        for task_name, task_obj in self.multitask_tasks.items():
            criterion.set_multitask_loss_weight(
                task_name, task_obj.args.get_loss_weight(update_num)
            )
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].train()

        loss, sample_size, logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )
        return loss, sample_size, logging_output

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

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        bpe_symbol = self.cfg.eval_bleu_remove_bpe

        def decode(toks, is_ref, prefix_size):
            unk_str = "UNKNOWNTOKENINREF" if is_ref else "UNKNOWNTOKENINHYP"
            s = self.target_dictionary.string(
                toks.long().cpu(), bpe_symbol, unk_string=unk_str
            )
            if prefix_size > 0:
                s = s.split(" ", maxsplit=prefix_size)
                s = s[prefix_size] if len(s) > prefix_size else ""
            return s

        prefix_tokens = None
        pfx_size = self.args.ignore_prefix_size
        if pfx_size > 0:
            prefix_tokens = sample["target"][:, :pfx_size]
        gen_out = self.inference_step(
            generator, [model], sample, prefix_tokens=prefix_tokens
        )
        hypo, ref = [], []
        pad = self.target_dictionary.pad()
        for i in range(len(gen_out)):
            hypo.append(decode(gen_out[i][0]["tokens"], False, pfx_size))
            ref.append(decode(strip_pad(sample["target"][i], pad), True, pfx_size))
        if self.cfg.eval_bleu_print_samples:
            if torch.distributed.get_rank() == 0:
                print("H-{} {}".format(sample["id"][0], hypo[0]))
                print("T-{} {}".format(sample["id"][0], ref[0]))

        eval_tokenizer = self.cfg.eval_bleu_tokenizer
        return sacrebleu.corpus_bleu(hypo, [ref], tokenize=eval_tokenizer)

    def valid_step(self, sample, model, criterion):
        for task_name, task_obj in self.multitask_tasks.items():
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].eval()

        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            score = self._inference_with_bleu(self.generator, sample, model)
            logging_output["_bleu_sys_len"] = score.sys_len
            logging_output["_bleu_ref_len"] = score.ref_len
            assert len(score.counts) == 4
            for i in range(4):
                logging_output[f"_bleu_counts_{i}"] = score.counts[i]
                logging_output[f"_bleu_totals_{i}"] = score.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:
            import sacrebleu

            len_keys = ["_bleu_sys_len", "_bleu_ref_len"]
            count_keys = [f"_bleu_counts_{i}" for i in range(4)]
            total_keys = [f"_bleu_totals_{i}" for i in range(4)]
            for k in len_keys + count_keys + total_keys:
                metrics.log_scalar(k, sum(x.get(k, 0) for x in logging_outputs))

            metrics.log_derived(
                "bleu",
                lambda meters: sacrebleu.BLEU.compute_bleu(
                    correct=[int(meters[k].sum) for k in count_keys],
                    total=[int(meters[k].sum) for k in total_keys],
                    sys_len=int(meters["_bleu_sys_len"].sum),
                    ref_len=int(meters["_bleu_ref_len"].sum),
                    smooth_method="exp",
                ).score,
            )


class DummyMultiTask(LegacyFairseqTask):
    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if self.args.decoder_type == "ctc":
            model = models[0]  # only support single model
            encoder_out = model(**sample)
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(
                    encoder_out
                )  # no need to normalize emissions
            else:
                emissions = model.get_normalized_probs(encoder_out, log_probs=True)
            return generator.decode(
                emissions.transpose(0, 1).float().cpu().contiguous()
            )
        else:
            raise NotImplementedError("only ctc decoder is supported at the moment")

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if self.args.decoder_type == "ctc":
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

            return W2lViterbiDecoder(args, self.tgt_dict)
        else:
            raise NotImplementedError("only ctc decoder is supported at the moment")
