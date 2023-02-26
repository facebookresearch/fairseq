# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import itertools
import logging
import os

import numpy as np
import torch

from fairseq.logging import metrics
from fairseq.data import (
    ConcatDataset,
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    indexed_dataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset,
    TokenBlockDataset,
)
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from omegaconf import II, MISSING


EVAL_BLEU_ORDER = 4
TARGET_METRIC_CHOICES = ChoiceEnum(["bleu", "ter"])

logger = logging.getLogger(__name__)


@dataclass
class DiscriminativeRerankingNMTConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    num_data_splits: int = field(
        default=1, metadata={"help": "total number of data splits"}
    )
    no_shuffle: bool = field(
        default=False, metadata={"help": "do not shuffle training data"}
    )
    max_positions: int = field(
        default=512, metadata={"help": "number of positional embeddings to learn"}
    )
    include_src: bool = field(
        default=False, metadata={"help": "include source sentence"}
    )
    mt_beam: int = field(default=50, metadata={"help": "beam size of input hypotheses"})
    eval_target_metric: bool = field(
        default=False,
        metadata={"help": "evaluation with the target metric during validation"},
    )
    target_metric: TARGET_METRIC_CHOICES = field(
        default="bleu", metadata={"help": "name of the target metric to optimize for"}
    )
    train_subset: str = field(
        default=II("dataset.train_subset"),
        metadata={"help": "data subset to use for training (e.g. train, valid, test)"},
    )
    seed: int = field(
        default=II("common.seed"),
        metadata={"help": "pseudo random number generator seed"},
    )


class RerankerScorer(object):
    """Scores the target for a given (source (optional), target) input."""

    def __init__(self, args, mt_beam):
        self.mt_beam = mt_beam

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]

        assert len(models) == 1, "does not support model ensemble"
        model = models[0]

        bs = net_input["src_tokens"].shape[0]
        assert (
            model.joint_classification == "none" or bs % self.mt_beam == 0
        ), f"invalid batch size ({bs}) for joint classification with beam size ({self.mt_beam})"

        model.eval()
        logits = model(**net_input)

        batch_out = model.sentence_forward(logits, net_input["src_tokens"])
        if model.joint_classification == "sent":
            batch_out = model.joint_forward(
                batch_out.view(self.mt_beam, bs // self.mt_beam, -1)
            )
        scores = model.classification_forward(
            batch_out.view(bs, 1, -1)
        )  # input: B x T x C

        return scores


@register_task(
    "discriminative_reranking_nmt", dataclass=DiscriminativeRerankingNMTConfig
)
class DiscriminativeRerankingNMTTask(FairseqTask):
    """
    Translation rerank task.
    The input can be either (src, tgt) sentence pairs or tgt sentence only.
    """

    cfg: DiscriminativeRerankingNMTConfig

    def __init__(self, cfg: DiscriminativeRerankingNMTConfig, data_dictionary=None):
        super().__init__(cfg)
        self.dictionary = data_dictionary
        self._max_positions = cfg.max_positions
        # args.tokens_per_sample = self._max_positions
        # self.num_classes = 1  # for model

    @classmethod
    def load_dictionary(cls, cfg, filename):
        """Load the dictionary from the filename"""
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")  # for loading pretrained XLMR model

        return dictionary

    @classmethod
    def setup_task(cls, cfg: DiscriminativeRerankingNMTConfig, **kwargs):
        # load data dictionary (assume joint dictionary)
        data_path = cfg.data
        data_dict = cls.load_dictionary(
            cfg, os.path.join(data_path, "input_src/dict.txt")
        )

        logger.info("[input] src dictionary: {} types".format(len(data_dict)))

        return DiscriminativeRerankingNMTTask(cfg, data_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        if self.cfg.data.endswith("1"):
            data_shard = (epoch - 1) % self.cfg.num_data_splits + 1
            data_path = self.cfg.data[:-1] + str(data_shard)
        else:
            data_path = self.cfg.data

        def get_path(type, data_split):
            return os.path.join(data_path, str(type), data_split)

        def make_dataset(type, dictionary, data_split, combine):
            split_path = get_path(type, data_split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                combine=combine,
            )
            return dataset

        def load_split(data_split, metric):
            input_src = None
            if self.cfg.include_src:
                input_src = make_dataset(
                    "input_src", self.dictionary, data_split, combine=False
                )
                assert input_src is not None, "could not find dataset: {}".format(
                    get_path("input_src", data_split)
                )

            input_tgt = make_dataset(
                "input_tgt", self.dictionary, data_split, combine=False
            )
            assert input_tgt is not None, "could not find dataset: {}".format(
                get_path("input_tgt", data_split)
            )

            label_path = f"{get_path(metric, data_split)}.{metric}"
            assert os.path.exists(label_path), f"could not find dataset: {label_path}"

            np_labels = np.loadtxt(label_path)
            if self.cfg.target_metric == "ter":
                np_labels = -np_labels
            label = RawLabelDataset(np_labels)

            return input_src, input_tgt, label

        src_datasets = []
        tgt_datasets = []
        label_datasets = []

        if split == self.cfg.train_subset:
            for k in itertools.count():
                split_k = "train" + (str(k) if k > 0 else "")
                prefix = os.path.join(data_path, "input_tgt", split_k)
                if not indexed_dataset.dataset_exists(prefix, impl=None):
                    if k > 0:
                        break
                    else:
                        raise FileNotFoundError(f"Dataset not found: {prefix}")
                input_src, input_tgt, label = load_split(
                    split_k, self.cfg.target_metric
                )
                src_datasets.append(input_src)
                tgt_datasets.append(input_tgt)
                label_datasets.append(label)
        else:
            input_src, input_tgt, label = load_split(split, self.cfg.target_metric)
            src_datasets.append(input_src)
            tgt_datasets.append(input_tgt)
            label_datasets.append(label)

        if len(tgt_datasets) == 1:
            input_tgt, label = tgt_datasets[0], label_datasets[0]
            if self.cfg.include_src:
                input_src = src_datasets[0]
        else:
            input_tgt = ConcatDataset(tgt_datasets)
            label = ConcatDataset(label_datasets)
            if self.cfg.include_src:
                input_src = ConcatDataset(src_datasets)

        input_tgt = TruncateDataset(input_tgt, self.cfg.max_positions)
        if self.cfg.include_src:
            input_src = PrependTokenDataset(input_src, self.dictionary.bos())
            input_src = TruncateDataset(input_src, self.cfg.max_positions)
            src_lengths = NumelDataset(input_src, reduce=False)
            src_tokens = ConcatSentencesDataset(input_src, input_tgt)
        else:
            src_tokens = PrependTokenDataset(input_tgt, self.dictionary.bos())
            src_lengths = NumelDataset(src_tokens, reduce=False)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": src_lengths,
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "target": label,
        }

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        assert (
            len(dataset) % self.cfg.mt_beam == 0
        ), "dataset size (%d) is not a multiple of beam size (%d)" % (
            len(dataset),
            self.cfg.mt_beam,
        )

        # no need to shuffle valid/test sets
        if not self.cfg.no_shuffle and split == self.cfg.train_subset:

            # need to keep all hypothese together
            start_idx = np.arange(0, len(dataset), self.cfg.mt_beam)
            with data_utils.numpy_seed(self.cfg.seed + epoch):
                np.random.shuffle(start_idx)

            idx = np.arange(0, self.cfg.mt_beam)
            shuffle = np.tile(idx, (len(start_idx), 1)).reshape(-1) + np.tile(
                start_idx, (self.cfg.mt_beam, 1)
            ).transpose().reshape(-1)

            dataset = SortDataset(
                dataset,
                sort_order=[shuffle],
            )

        logger.info(f"Loaded {split} with #samples: {len(dataset)}")

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        assert not self.cfg.include_src or len(src_tokens[0]) == 2
        input_src = None
        if self.cfg.include_src:
            input_src = TokenBlockDataset(
                [t[0] for t in src_tokens],
                [l[0] for l in src_lengths],
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            )
            input_src = PrependTokenDataset(input_src, self.dictionary.bos())
            input_src = TruncateDataset(input_src, self.cfg.max_positions)

        input_tgt = TokenBlockDataset(
            [t[-1] for t in src_tokens],
            [l[-1] for l in src_lengths],
            block_size=None,  # ignored for "eos" break mode
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode="eos",
        )
        input_tgt = TruncateDataset(input_tgt, self.cfg.max_positions)
        if self.cfg.include_src:
            src_tokens = ConcatSentencesDataset(input_src, input_tgt)
            src_lengths = NumelDataset(input_src, reduce=False)
        else:
            input_tgt = PrependTokenDataset(input_tgt, self.dictionary.bos())
            src_tokens = input_tgt
            src_lengths = NumelDataset(src_tokens, reduce=False)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": src_lengths,
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        return NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

    def build_model(self, cfg: FairseqDataclass, from_checkpoint: bool = False):
        return super().build_model(cfg)

    def build_generator(self, args):
        return RerankerScorer(args, mt_beam=self.cfg.mt_beam)

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def create_dummy_batch(self, device):
        dummy_target = (
            torch.zeros(self.cfg.mt_beam, EVAL_BLEU_ORDER * 2 + 3).long().to(device)
            if not self.cfg.eval_ter
            else torch.zeros(self.cfg.mt_beam, 3).long().to(device)
        )

        return {
            "id": torch.zeros(self.cfg.mt_beam, 1).long().to(device),
            "net_input": {
                "src_tokens": torch.zeros(self.cfg.mt_beam, 4).long().to(device),
                "src_lengths": torch.ones(self.cfg.mt_beam, 1).long().to(device),
            },
            "nsentences": 0,
            "ntokens": 0,
            "target": dummy_target,
        }

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        if ignore_grad and sample is None:
            sample = self.create_dummy_batch(model.device)

        return super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )

    def valid_step(self, sample, model, criterion):
        if sample is None:
            sample = self.create_dummy_batch(model.device)

        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if not self.cfg.eval_target_metric:
            return loss, sample_size, logging_output

        scores = logging_output["scores"]

        if self.cfg.target_metric == "bleu":
            assert sample["target"].shape[1] == EVAL_BLEU_ORDER * 2 + 3, (
                "target does not contain enough information ("
                + str(sample["target"].shape[1])
                + "for evaluating BLEU"
            )

            max_id = torch.argmax(scores, dim=1)
            select_id = max_id + torch.arange(
                0, sample_size * self.cfg.mt_beam, self.cfg.mt_beam
            ).to(max_id.device)
            bleu_data = sample["target"][select_id, 1:].sum(0).data

            logging_output["_bleu_sys_len"] = bleu_data[0]
            logging_output["_bleu_ref_len"] = bleu_data[1]

            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu_data[2 + i]
                logging_output["_bleu_totals_" + str(i)] = bleu_data[
                    2 + EVAL_BLEU_ORDER + i
                ]

        elif self.cfg.target_metric == "ter":
            assert sample["target"].shape[1] == 3, (
                "target does not contain enough information ("
                + str(sample["target"].shape[1])
                + "for evaluating TER"
            )

            max_id = torch.argmax(scores, dim=1)
            select_id = max_id + torch.arange(
                0, sample_size * self.cfg.mt_beam, self.cfg.mt_beam
            ).to(max_id.device)
            ter_data = sample["target"][select_id, 1:].sum(0).data

            logging_output["_ter_num_edits"] = -ter_data[0]
            logging_output["_ter_ref_len"] = -ter_data[1]

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if not self.cfg.eval_target_metric:
            return

        def sum_logs(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        if self.cfg.target_metric == "bleu":
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
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)
        elif self.cfg.target_metric == "ter":
            num_edits = sum_logs("_ter_num_edits")
            ref_len = sum_logs("_ter_ref_len")

            if ref_len > 0:
                metrics.log_scalar("_ter_num_edits", num_edits)
                metrics.log_scalar("_ter_ref_len", ref_len)

                def compute_ter(meters):
                    score = meters["_ter_num_edits"].sum / meters["_ter_ref_len"].sum
                    return round(score.item(), 2)

                metrics.log_derived("ter", compute_ter)
