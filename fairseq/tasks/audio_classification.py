# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import itertools
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from omegaconf import II, MISSING
from sklearn import metrics as sklearn_metrics

from fairseq.data import AddTargetDataset, Dictionary, FileAudioDataset
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from fairseq.data.text_compressor import TextCompressionLevel, TextCompressor
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig, AudioPretrainingTask
from fairseq.tasks.audio_finetuning import label_len_fn, LabelEncoder

from .. import utils
from ..logging import metrics
from . import FairseqTask, register_task

logger = logging.getLogger(__name__)

@dataclass
class AudioClassificationConfig(AudioPretrainingConfig):
    target_dictionary: Optional[str] = field(
        default=None, metadata={"help": "override default dictionary location"}
    )


@register_task("audio_classification", dataclass=AudioClassificationConfig)
class AudioClassificationTask(AudioPretrainingTask):
    """Task for audio classification tasks."""

    cfg: AudioClassificationConfig

    def __init__(
        self,
        cfg: AudioClassificationConfig,
    ):
        super().__init__(cfg)
        self.state.add_factory("target_dictionary", self.load_target_dictionary)
        logging.info(f"=== Number of labels = {len(self.target_dictionary)}")

    def load_target_dictionary(self):
        if self.cfg.labels:
            target_dictionary = self.cfg.data
            if self.cfg.target_dictionary:  # override dict
                target_dictionary = self.cfg.target_dictionary
            dict_path = os.path.join(target_dictionary, f"dict.{self.cfg.labels}.txt")
            logger.info("Using dict_path : {}".format(dict_path))
            return Dictionary.load(dict_path, add_special_symbols=False)
        return None

    def load_dataset(
        self, split: str, task_cfg: AudioClassificationConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)
        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        data_path = self.cfg.data
        if task_cfg.multi_corpus_keys is None:
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
            text_compressor = TextCompressor(level=text_compression_level)
            with open(label_path, "r") as f:
                labels = [
                    text_compressor.compress(l)
                    for i, l in enumerate(f)
                    if i not in skipped_indices
                ]

            assert len(labels) == len(self.datasets[split]), (
                f"labels length ({len(labels)}) and dataset length "
                f"({len(self.datasets[split])}) do not match"
            )

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                label_len_fn=label_len_fn,
                add_to_input=False,
                # text_compression_level=text_compression_level,
            )
        else:
            target_dataset_map = OrderedDict()

            multi_corpus_keys = [
                k.strip() for k in task_cfg.multi_corpus_keys.split(",")
            ]
            corpus_idx_map = {k: idx for idx, k in enumerate(multi_corpus_keys)}

            data_keys = [k.split(":") for k in split.split(",")]

            multi_corpus_sampling_weights = [
                float(val.strip())
                for val in task_cfg.multi_corpus_sampling_weights.split(",")
            ]
            data_weights = []
            for key, file_name in data_keys:
                k = key.strip()
                label_path = os.path.join(
                    data_path, f"{file_name.strip()}.{task_cfg.labels}"
                )
                skipped_indices = getattr(
                    self.dataset_map[split][k], "skipped_indices", set()
                )
                text_compressor = TextCompressor(level=text_compression_level)
                with open(label_path, "r") as f:
                    labels = [
                        text_compressor.compress(l)
                        for i, l in enumerate(f)
                        if i not in skipped_indices
                    ]

                assert len(labels) == len(self.dataset_map[split][k]), (
                    f"labels length ({len(labels)}) and dataset length "
                    f"({len(self.dataset_map[split][k])}) do not match"
                )

                process_label = LabelEncoder(self.target_dictionary)

                # TODO: Remove duplication of code from the if block above
                target_dataset_map[k] = AddTargetDataset(
                    self.dataset_map[split][k],
                    labels,
                    pad=self.target_dictionary.pad(),
                    eos=self.target_dictionary.eos(),
                    batch_targets=True,
                    process_label=process_label,
                    label_len_fn=label_len_fn,
                    add_to_input=False,
                    # text_compression_level=text_compression_level,
                )

                data_weights.append(multi_corpus_sampling_weights[corpus_idx_map[k]])

            if len(target_dataset_map) == 1:
                self.datasets[split] = list(target_dataset_map.values())[0]
            else:
                self.datasets[split] = MultiCorpusDataset(
                    target_dataset_map,
                    distribution=data_weights,
                    seed=0,
                    sort_indices=True,
                )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def train_step(self, sample, model, *args, **kwargs):
        sample["target"] = sample["target"].to(dtype=torch.long)
        loss, sample_size, logging_output = super().train_step(
            sample, model, *args, **kwargs
        )
        self._log_metrics(sample, model, logging_output)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        sample["target"] = sample["target"].to(dtype=torch.long)
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        self._log_metrics(sample, model, logging_output)
        return loss, sample_size, logging_output

    def _log_metrics(self, sample, model, logging_output):
        metrics = self._inference_with_metrics(
            sample,
            model,
        )
        """
        logging_output["_precision"] = metrics["precision"]
        logging_output["_recall"] = metrics["recall"]
        logging_output["_f1"] = metrics["f1"]
        logging_output["_eer"] = metrics["eer"]
        logging_output["_accuracy"] = metrics["accuracy"]
        """
        logging_output["_correct"] = metrics["correct"]
        logging_output["_total"] = metrics["total"]

    def _inference_with_metrics(self, sample, model):
        def _compute_eer(target_list, lprobs):
            # from scipy.optimize import brentq
            # from scipy.interpolate import interp1d

            y_one_hot = np.eye(len(self.state.target_dictionary))[target_list]
            fpr, tpr, thresholds = sklearn_metrics.roc_curve(
                y_one_hot.ravel(), lprobs.ravel()
            )
            # Revisit the interpolation approach.
            # eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

            fnr = 1 - tpr
            eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

            return eer

        with torch.no_grad():
            net_output = model(**sample["net_input"])
            lprobs = (
                model.get_normalized_probs(net_output, log_probs=True).cpu().detach()
            )
            target_list = sample["target"][:, 0].detach().cpu()
            predicted_list = torch.argmax(lprobs, 1).detach().cpu()  # B,C->B

            metrics = {
                "correct": torch.sum(target_list == predicted_list).item(),
                "total": len(target_list),
            }
            return metrics

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        correct, total = 0, 0
        for log in logging_outputs:
            correct += log.get("_correct", zero)
            total += log.get("_total", zero)
        metrics.log_scalar("_correct", correct)
        metrics.log_scalar("_total", total)

        if total > 0:
            def _fn_accuracy(meters):
                if meters["_total"].sum > 0:
                    return utils.item(meters["_correct"].sum / meters["_total"].sum)
                return float("nan")

            metrics.log_derived("accuracy", _fn_accuracy)
        """
        prec_sum, recall_sum, f1_sum, acc_sum, eer_sum = 0.0, 0.0, 0.0, 0.0, 0.0
        for log in logging_outputs:
            prec_sum += log.get("_precision", zero).item()
            recall_sum += log.get("_recall", zero).item()
            f1_sum += log.get("_f1", zero).item()
            acc_sum += log.get("_accuracy", zero).item()
            eer_sum += log.get("_eer", zero).item()

        metrics.log_scalar("avg_precision", prec_sum / len(logging_outputs))
        metrics.log_scalar("avg_recall", recall_sum / len(logging_outputs))
        metrics.log_scalar("avg_f1", f1_sum / len(logging_outputs))
        metrics.log_scalar("avg_accuracy", acc_sum / len(logging_outputs))
        metrics.log_scalar("avg_eer", eer_sum / len(logging_outputs))
        """