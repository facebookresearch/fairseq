# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import math
import os
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import numpy as np
import torch

from examples.laser.laser_src.multitask_data_utils import (
    MultidatasetEpochBatchIterator,
    MultitaskDatasetWrapper,
)
from fairseq import metrics, models, options, utils
from fairseq.criterions.masked_lm import MaskedLmLoss
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    FairseqDataset,
    IndexedDataset,
    LanguagePairDataset,
    MaskTokensDataset,
    PrependTokenDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.tasks import LegacyFairseqTask, register_task, setup_task

logger = logging.getLogger(__name__)


class TaskLanguageConfig(object):
    def __init__(self, task, source, target):
        assert task in ["self", "mask", "distil", "tlm"]
        self.task = task
        self.source = source
        self.target = target

    @classmethod
    def parse(cls, student_teacher_task):
        task, source_target = student_teacher_task.split(":")
        source, target = source_target.split("-")

        return TaskLanguageConfig(task, source, target)

    def get_key(self):
        return f"{self.task}:{self.source}-{self.target}"


@register_task("laser_distillation")
class LaserDistillationTask(LegacyFairseqTask):  # TODO: move to FairseqTask
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "configfile", metavar="PATH", help="dataset configuration file in json"
        )
        parser.add_argument(
            "--joint-weighting-alpha",
            type=float,
            default=0.0,
            help="alpha for automatic weighting of both distillation+MLM tasks",
        )
        parser.add_argument(
            "--distil-weighting-alpha",
            type=float,
            default=0.0,
            help="alpha for automatic weighting of distillation tasks",
        )
        parser.add_argument(
            "--mask-weighting-alpha",
            type=float,
            default=0.0,
            help="alpha for automatic weighting of MLM tasks",
        )
        parser.add_argument(
            "--ignore-original-sampling",
            action="store_true",
            help="ignore sample value in the config json",
        )
        parser.add_argument(
            "--raw-text", action="store_true", help="load raw text dataset"
        )
        parser.add_argument(
            "--left-pad-source",
            default="True",
            type=str,
            metavar="BOOL",
            help="pad the source on the left (default: True)",
        )
        parser.add_argument(
            "--left-pad-target",
            default="False",
            type=str,
            metavar="BOOL",
            help="pad the target on the left (default: False)",
        )
        parser.add_argument(
            "--teacher-checkpoint-path",
            type=str,
            required=True,
            metavar="STR",
            help="path to pre-trained teacher",
        )
        parser.add_argument(
            "--debug-size", action="store_true", help="use only a part of the dataset"
        )
        parser.add_argument(
            "--debug-init-student",
            action="store_true",
            help="initialize the student as well",
        )
        parser.add_argument(
            "--lambda-self", required=True, type=float, help="self loss coefficient"
        )
        parser.add_argument(
            "--lambda-mask", required=True, type=float, help="mask loss coefficient"
        )
        parser.add_argument(
            "--lambda-distil", required=True, type=float, help="distil loss coefficient"
        )

        parser.add_argument(
            "--do-permutation",
            action="store_true",
            help="keep the random permutation on repetition for oversampling",
        )

        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )
        parser.add_argument(
            "--mask-whole-words",
            default=False,
            action="store_true",
            help="mask whole words; you may also want to set --bpe",
        )

        parser.add_argument(
            "--student-teacher-config",
            type=str,
            default="",
            metavar="STR",
            help="filter to specific languages as source and target",
        )
        parser.add_argument(
            "--prepend-bos",
            action="store_true",
            help="Add Bos token at the beginning of source sentences",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )
        parser.add_argument(
            "--student-checkpoint-path",
            default=None,
            type=str,
            help="path to student checkpoint model",
        )

    def __init__(self, args, config, src_dictionary, tgt_dictionary, num_tasks):
        super().__init__(args)
        self.config = config
        self.src_dictionary = src_dictionary
        self.tgt_dictionary = tgt_dictionary
        self.num_tasks = num_tasks
        self.sample_print = SamplePrint(
            src_dictionary, tgt_dictionary, interval=1000, samples=5
        )
        # added to dictionary during setup_task
        self.mask_idx = self.src_dictionary.index("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        config = json.load(open(args.configfile))
        num_tasks = max([dataset["id"] for dataset in config["train"]]) + 1

        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        assert (
            config["src_vocab"] and config["tgt_vocab"]
        ), f"Source and target vocab must be specified"

        src_dictionary = Dictionary.load(config["src_vocab"])
        src_dictionary.add_symbol("<mask>")
        tgt_dictionary = Dictionary.load(config["tgt_vocab"])

        logger.info(
            "src dictionary {} : {} types".format(
                config["src_vocab"], len(src_dictionary)
            )
        )
        logger.info(
            "tgt dictionary {} : {} types".format(
                config["tgt_vocab"], len(tgt_dictionary)
            )
        )

        return cls(args, config, src_dictionary, tgt_dictionary, num_tasks)

    def build_model(self, args):
        student_model = models.build_model(args, self)
        # initialise student using checkpoint
        if (
            hasattr(args, "student_checkpoint_path")
            and args.student_checkpoint_path is not None
        ):
            logger.info(
                f"initialising student using checkpoint: {args.student_checkpoint_path}"
            )
            student_checkpoint = torch.load(args.student_checkpoint_path)
            student_encoder_model_checkpoint = get_encoder_model_checkpoint(
                student_checkpoint["model"]
            )
            student_model.encoder.load_state_dict(
                student_encoder_model_checkpoint, strict=False
            )

        teacher_checkpoint = torch.load(args.teacher_checkpoint_path)
        if teacher_checkpoint["args"]:
            teacher_args = teacher_checkpoint["args"]
        else:
            teacher_args = teacher_checkpoint["cfg"]["model"]
        teacher_args.configfile = args.configfile

        teacher_task = setup_task(teacher_args)

        # ensure that the teacher's encoder uses the vocab specified in tgt
        teacher_task.src_dictionary = teacher_task.tgt_dictionary

        teacher_model = teacher_task.build_model(teacher_args)

        with check_before_after_modelsize(teacher_model):
            teacher_encoder_model_checkpoint = get_encoder_model_checkpoint(
                teacher_checkpoint["model"]
            )
            if hasattr(teacher_model.encoder, "fc_out"):
                del teacher_model.encoder.fc_out

            teacher_model.encoder.load_state_dict(
                teacher_encoder_model_checkpoint, strict=True
            )

        student_teacher_tasks = []
        for student_teacher_task in args.student_teacher_config.split(","):
            student_teacher_tasks.append(TaskLanguageConfig.parse(student_teacher_task))
        self.student_teacher_tasks = student_teacher_tasks

        if args.fp16:
            teacher_model = teacher_model.half()

        teacher_model.eval()
        self.teacher_model = teacher_model

        for p in student_model.decoder.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
            p.requires_grad = False

        logging.info(f"student encoder: {student_model.encoder}")
        logging.info(f"teacher encoder: {teacher_model.encoder}")

        return student_model

    def dataset(self, split):
        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        return self.datasets[split]

    def get_config_for_student_teacher_langs(self, split, student_teacher_task):
        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                raise Exception("Unable to handle raw text.")
            dataset = IndexedDataset(path, fix_lua_indexing=True)
            return dataset

        datasets = []
        for dataset_config in self.config[split]:
            src_path = os.path.dirname(dataset_config["src"])
            corpus_name = src_path.split("/")[-2]
            language_pair_name = src_path.split("/")[-1]
            pair_datasets_key = corpus_name + "__" + language_pair_name
            language_pair_name = language_pair_name.split("-")

            if len(language_pair_name) == 2:
                dataset_lang_src, dataset_lang_tgt = language_pair_name
            elif len(language_pair_name) == 1:
                dataset_lang_src = language_pair_name[0]
                dataset_lang_tgt = language_pair_name[0]

            should_add_this = False

            if (
                dataset_lang_src == student_teacher_task.source
                and dataset_lang_tgt == student_teacher_task.target
            ):
                should_add_this = True

            if should_add_this:
                if student_teacher_task.task in ["mask", "tlm"]:
                    assert self.args.arch != "laser_lstm" or getattr(
                        self.args, "vocab_out", False
                    ), f"You should set --vocab-out for laser_lstm for masking"

                    untouched_src_dataset = indexed_dataset(
                        dataset_config["src"], self.source_dictionary
                    )
                    # TLM
                    if student_teacher_task.task == "tlm":
                        untouched_tgt_dataset = indexed_dataset(
                            dataset_config["tgt"],
                            self.source_dictionary,  # note using student dictionary
                        )
                        dataset = ConcatSentencesDataset(
                            untouched_src_dataset, untouched_tgt_dataset
                        )
                    else:
                        # create continuous blocks of tokens
                        dataset = TokenBlockDataset(
                            untouched_src_dataset,
                            untouched_src_dataset.sizes,
                            self.args.tokens_per_sample - 1,  # one less for <s>
                            pad=self.source_dictionary.pad(),
                            eos=self.source_dictionary.eos(),
                            break_mode=self.args.sample_break_mode,
                        )
                        logger.info("loaded {} blocks".format(len(dataset)))

                    src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
                        dataset,
                        self.source_dictionary,
                        pad_idx=self.source_dictionary.pad(),
                        mask_idx=self.mask_idx,
                        seed=self.args.seed,
                        mask_prob=self.args.mask_prob,
                        leave_unmasked_prob=self.args.leave_unmasked_prob,
                        random_token_prob=self.args.random_token_prob,
                        freq_weighted_replacement=self.args.freq_weighted_replacement,
                        mask_whole_words=None,
                    )

                    # prepend CLS token (done after masking to ensure CLS isn't masked)
                    src_dataset = PrependTokenDataset(
                        src_dataset, self.source_dictionary.bos()
                    )
                    tgt_dataset = PrependTokenDataset(
                        tgt_dataset, self.source_dictionary.pad()
                    )

                    dataset = LanguagePairDataset(
                        src_dataset,
                        src_dataset.sizes,
                        self.source_dictionary,
                        tgt_dataset,
                        tgt_dataset.sizes,
                        self.source_dictionary,
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_target,
                    )
                elif student_teacher_task.task == "self":
                    # student and teacher share same 'src' in config and associated source_dictionary
                    tgt_dataset = indexed_dataset(
                        dataset_config["src"], self.source_dictionary
                    )
                    # only pre-append the source dataset
                    if self.args.prepend_bos:
                        src_dataset = PrependTokenDataset(
                            tgt_dataset, self.source_dictionary.bos()
                        )
                    else:
                        src_dataset = tgt_dataset

                    dataset = LanguagePairDataset(
                        src_dataset,
                        src_dataset.sizes,
                        self.source_dictionary,
                        tgt_dataset,
                        tgt_dataset.sizes,
                        self.source_dictionary,
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_source,
                    )
                else:
                    src_dataset = indexed_dataset(
                        dataset_config["src"], self.source_dictionary
                    )
                    tgt_dataset = indexed_dataset(
                        dataset_config["tgt"], self.target_dictionary
                    )
                    if self.args.prepend_bos:
                        src_dataset = PrependTokenDataset(
                            src_dataset, self.source_dictionary.bos()
                        )

                    dataset = LanguagePairDataset(
                        src_dataset,
                        src_dataset.sizes,
                        self.source_dictionary,
                        tgt_dataset,
                        tgt_dataset.sizes,
                        self.target_dictionary,
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_target,
                    )

                datasets.append(
                    {
                        "task_key": student_teacher_task.get_key(),
                        "key": pair_datasets_key,
                        "dataset": dataset,
                        "target_id": dataset_config["id"],
                        "original_sample": dataset_config.get("sample", 1.0)
                        if not self.args.ignore_original_sampling
                        else 1.0,
                    }
                )

        if len(datasets) == 0:
            raise RuntimeError(f"Missing data for {student_teacher_task.get_key()}")
        return datasets

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a dataset split."""

        pair_datasets = OrderedDict()

        if split == "valid":
            self.datasets[split] = pair_datasets
            return

        if split not in self.config:
            raise FileNotFoundError(
                "Dataset not found in config file: {}".format(split)
            )

        lang_datasets = {}
        for student_teacher_task in self.student_teacher_tasks:
            lang = student_teacher_task.source
            task_lang = lang
            if not self.args.joint_weighting_alpha:
                task = student_teacher_task.task
                # for purposes of upsampling treat self+distil as one
                if task == "self":
                    task = "distil"
                elif task == "tlm":
                    task = "mask"
                task_lang = task + "_" + lang
            datasets = self.get_config_for_student_teacher_langs(
                split, student_teacher_task
            )
            sum_dataset_lengths = sum([len(d["dataset"]) for d in datasets])

            if task_lang in lang_datasets:
                lang_datasets[task_lang]["datasets"] += datasets
                lang_datasets[task_lang]["len"] += sum_dataset_lengths
            else:
                lang_datasets[task_lang] = {
                    "datasets": datasets,
                    "len": sum_dataset_lengths,
                }

        # returns (task-srclang, upsample weight)
        if self.args.joint_weighting_alpha:
            weighted_langs = compute_weighting_joint(
                lang_datasets, self.args.joint_weighting_alpha
            )
        else:
            weighted_langs = compute_weighting_separate(lang_datasets, self.args)

        dataset_task_ctr = 0
        multitask_datasets = {}
        for task_lang in lang_datasets:
            multitask_datasets = []
            total_size_for_student_teacher_pair = 0

            for dataset_cfg in lang_datasets[task_lang]["datasets"]:
                dataset_task_ctr += 1
                dataset = dataset_cfg["dataset"]
                task_key = dataset_cfg["task_key"]
                added_pair_datasets_key = (
                    f"{dataset_task_ctr}__{task_key}__{dataset_cfg['key']}"
                )
                # the multitask dataset wrapper contains the logic to upsample when loading indices
                multitask_dataset = MultitaskDatasetWrapper(
                    dataset=dataset,
                    target_language_id=dataset_cfg["target_id"],
                    sample=dataset_cfg["original_sample"] * weighted_langs[task_lang],
                    do_permutation=self.args.do_permutation,
                    name=added_pair_datasets_key,
                )
                multitask_datasets.append(multitask_dataset)
                pair_datasets[added_pair_datasets_key] = multitask_dataset
                total_size_for_student_teacher_pair += len(multitask_dataset)

            lang_datasets[task_lang]["multi_datasets"] = multitask_datasets

        # display
        logger.info(f"Upsampled datasets:")
        for task_lang in lang_datasets:
            datasets = lang_datasets[task_lang]
            lwv = sum([len(d) for d in datasets["multi_datasets"]])

            logger.info(
                f"task_lang = {task_lang} ({datasets['len']}). ({lwv} weighted)"
            )
            for dataset in datasets["datasets"]:
                logger.info(f"\t\t{dataset['key']}\t\t{len(dataset['dataset'])}")
            for dataset in datasets["multi_datasets"]:
                logger.info(f"\t\t{dataset.name}\t{len(dataset)} (weighted)")

        self.datasets[split] = pair_datasets

    def build_criterion(self, args):
        ret = super().build_criterion(args)

        masked_lm_criterion = MaskedLmLoss(args, task=self)
        self.masked_lm_criterion = masked_lm_criterion

        return ret

    @property
    def source_dictionary(self):
        return self.src_dictionary

    @property
    def target_dictionary(self):
        return self.tgt_dictionary

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):

        assert isinstance(dataset, OrderedDict)
        assert len(dataset)
        assert isinstance(dataset[next(iter(dataset))], FairseqDataset)

        # initialize the dataset with the correct starting epoch
        for key, dt in dataset.items():
            dt.set_epoch(epoch)

        batch_sampler = OrderedDict()

        with data_utils.numpy_seed(seed + epoch):
            indices_key = []
            for key, dt in dataset.items():
                indices_key = dt.ordered_indices()

                # filter examples that are too large
                if max_positions is not None:
                    indices_key, ignored = dt.filter_indices_by_size(
                        indices_key, max_positions
                    )

                batch_sampler[key] = data_utils.batch_by_size(
                    indices_key,
                    dt.num_tokens,
                    max_tokens=max_tokens,
                    max_sentences=max_sentences,
                    required_batch_size_multiple=required_batch_size_multiple,
                )

                del indices_key

        epoch_iter = MultidatasetEpochBatchIterator(
            dataset=dataset,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )

        return epoch_iter

    # ignore_grad used for dummy batches
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)

        model.train()
        model.set_num_updates(update_num)

        model_device = next(model.parameters()).device
        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device != model_device:
            self.teacher_model.to(model_device)

        weights = {
            "self": self.args.lambda_self,
            "mask": self.args.lambda_mask,
            "distil": self.args.lambda_distil,
            "tlm": self.args.lambda_mask,
        }

        def forward_backward(model, smp, teacher_order, weight_key):
            nonlocal agg_loss, agg_sample_size, agg_logging_output

            with torch.autograd.profiler.record_function("forward"):
                if weight_key in ["mask", "tlm"]:
                    loss, sample_size, logging_output = self.masked_lm_criterion(
                        model.encoder, smp, teacher_order
                    )
                else:
                    loss, sample_size, logging_output = criterion(
                        model, smp, teacher_order
                    )
            if ignore_grad or smp["is_dummy"]:
                loss *= 0
            else:
                loss *= weights[weight_key]
            optimizer.backward(loss)

            agg_loss += loss.detach().item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[weight_key + "_" + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]

        net_input = sample["net_input"]
        dataset_name = net_input["dataset_name"]
        dataset_name_args = dataset_name.split("__")
        student_teacher_task_key = dataset_name_args[1]
        student_teacher_task = TaskLanguageConfig.parse(student_teacher_task_key)
        teacher_enc_out = None

        with torch.no_grad():
            teacher_src_tokens = sample["target"]
            teacher_src_lengths = (
                (teacher_src_tokens.ne(self.target_dictionary.pad())).long().sum(dim=1)
            )

            teacher_order = torch.argsort(teacher_src_lengths, descending=True)
            teacher_order = teacher_order[teacher_src_lengths[teacher_order] > 0]

            teacher_src_tokens = teacher_src_tokens[teacher_order]
            teacher_src_lengths = teacher_src_lengths[teacher_order]

            if teacher_order.size(0) == 0:
                logger.warning(f"skipped sample. lengths = {teacher_src_lengths}")
                return agg_loss, agg_sample_size, agg_logging_output

            if student_teacher_task.task not in ["mask", "tlm"]:
                max_size = teacher_src_lengths.detach().max().item()
                teacher_src_tokens = teacher_src_tokens[:, :max_size]
                teacher_enc_out = self.teacher_model.encoder.forward(
                    teacher_src_tokens, teacher_src_lengths, ""
                )

        if teacher_enc_out:
            sample["teacher_enc_out"] = teacher_enc_out

        if student_teacher_task.task in ["mask", "tlm"]:
            self.sample_print(
                net_input["src_tokens"],
                sample["target"],
                student_teacher_task,
            )
        else:
            self.sample_print(
                net_input["src_tokens"][teacher_order],
                teacher_src_tokens,
                student_teacher_task,
            )

        forward_backward(model, sample, teacher_order, student_teacher_task.task)

        return agg_loss, agg_sample_size, agg_logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        for spec_key in ["mask", "self", "distil", "tlm"]:
            sample_size = sum(
                x.get(spec_key + "_sample_size", 0) for x in logging_outputs
            )
            if sample_size:
                loss_sum = sum(x.get(spec_key + "_loss", 0) for x in logging_outputs)
                loss_sum *= 1 / sample_size
                metrics.log_scalar(spec_key + "_loss", loss_sum, sample_size, round=3)

                if spec_key == "mask":
                    metrics.log_derived(
                        "mask_ppl",
                        lambda meters: utils.get_perplexity(meters["mask_loss"].avg),
                    )
                if spec_key == "tlm":
                    metrics.log_derived(
                        "tlm_ppl",
                        lambda meters: utils.get_perplexity(meters["tlm_loss"].avg),
                    )


class SamplePrint:
    def __init__(self, source_dictionary, target_dictionary, interval, samples):
        self.source_dictionary = source_dictionary
        self.target_dictionary = target_dictionary
        self.interval = interval
        self.samples = samples
        self.counter = 1

    def __call__(self, student_src_tokens, teacher_src_tokens, student_teacher_task):
        self.counter += 1
        if self.counter < self.interval:
            return
        if self.counter > self.interval:
            self.counter = 0

        student_lang = student_teacher_task.source
        if student_teacher_task.task in ["self", "mask", "tlm"]:
            teacher_lang = student_lang
        else:
            teacher_lang = student_teacher_task.target

        ln = student_src_tokens.shape[0]

        for i in range(min(ln, self.samples)):
            src_str = self.source_dictionary.string(
                student_src_tokens[i], "sentencepiece"
            )
            if student_teacher_task.task in ["mask", "tlm"]:
                src_str = self.source_dictionary.string(
                    student_src_tokens[i],
                )
                tgt_str = self.source_dictionary.string(
                    teacher_src_tokens[i],
                )
            else:
                tgt_str = self.target_dictionary.string(
                    teacher_src_tokens[i], "sentencepiece"
                )
            logger.info(
                "\n{}\t\t[{} student ]  {}\n\t\t[{} student ]  {}\n\t\t[{} teacher ]  {}\n\t\t[{} teacher ]  {}\n\t\t\n".format(
                    i,
                    student_lang,
                    src_str,
                    student_lang,
                    student_src_tokens[i],
                    teacher_lang,
                    tgt_str,
                    teacher_lang,
                    teacher_src_tokens[i],
                )
            )


@contextmanager
def check_before_after_modelsize(model):
    def get_model_size(caption, model):
        w_size = 0.0
        nb_params = 0
        for p in model.parameters():
            w_size += np.sum(p.cpu().data.numpy())
            nb_params += p.numel()

        logger.info(f"{caption:8} nb_param = {nb_params} size = {w_size}")
        return w_size

    before = get_model_size("before", model)
    yield before
    after = get_model_size("after", model)
    assert before != after


def get_encoder_model_checkpoint(state_dict):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if key.startswith("encoder"):
            new_key = key[len("encoder.") :]
            new_state_dict[new_key] = state_dict[key]
    return new_state_dict


# compute weighting per lang
def compute_weighting_joint(dataset_dict, weighting_alpha):
    dataset_dict_samples = {}
    dataset_dict_frq_weights = {}
    length_sum = 0
    weighted_freqs_sum = 0
    freq_per_dataset = {}
    vmax = 0
    vmin = 1
    weighted_freq_per_dataset = {}

    for key in dataset_dict:
        length_sum += float(dataset_dict[key]["len"])

    for key in dataset_dict:
        val = float(dataset_dict[key]["len"]) / length_sum
        freq_per_dataset[key] = val
        weighted_freqs_sum += val**weighting_alpha

    for key in freq_per_dataset:
        val = freq_per_dataset[key] ** weighting_alpha / weighted_freqs_sum
        vmin = min(vmin, val)
        vmax = max(vmax, val)
        dataset_dict_frq_weights[key] = val

    for key in freq_per_dataset:
        if dataset_dict_frq_weights[key] == 0:
            logger.warning(f"0 samples for {key}")
            dataset_dict_samples[key] = 0
        else:
            dataset_dict_samples[key] = vmax / dataset_dict_frq_weights[key]

    return dataset_dict_samples


# compute weighting per task per lang
def compute_weighting_separate(dataset_dict, args):
    tasks = ["distil", "mask"]

    def initialise(keys, val):
        dict = {}
        for key in keys:
            dict[key] = val
        return dict

    dataset_dict_samples = {}
    dataset_dict_frq_weights = {}
    freq_per_dataset = {}
    length_sum = initialise(tasks, 0)
    weighted_freqs_sum = initialise(tasks, 0)
    vmax = initialise(tasks, 0)

    for key in dataset_dict:
        task = key.split("_")[0]
        length_sum[task] += float(dataset_dict[key]["len"])

    for key in dataset_dict:
        task = key.split("_")[0]
        weighting_alpha = (
            args.distil_weighting_alpha
            if task == "distil"
            else args.mask_weighting_alpha
        )
        val = float(dataset_dict[key]["len"]) / length_sum[task]
        freq_per_dataset[key] = val
        weighted_freqs_sum[task] += val**weighting_alpha

    for key in freq_per_dataset:
        task = key.split("_")[0]
        weighting_alpha = (
            args.distil_weighting_alpha
            if task == "distil"
            else args.mask_weighting_alpha
        )
        val = freq_per_dataset[key] ** weighting_alpha / weighted_freqs_sum[task]
        vmax[task] = max(vmax[task], val)
        dataset_dict_frq_weights[key] = val

    for key in freq_per_dataset:
        task = key.split("_")[0]
        if dataset_dict_frq_weights[key] == 0:
            logger.warning(f"0 samples for {key}")
            dataset_dict_samples[key] = 0
        else:
            dataset_dict_samples[key] = vmax[task] / dataset_dict_frq_weights[key]

    return dataset_dict_samples
