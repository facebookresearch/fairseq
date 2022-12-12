# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import numpy as np
import math
import torch

from sklearn import metrics as sklearn_metrics
from dataclasses import dataclass

from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.tasks import register_task
from fairseq.logging import metrics

from ..data.add_class_target_dataset import AddClassTargetDataset


logger = logging.getLogger(__name__)


@dataclass
class AudioClassificationConfig(AudioPretrainingConfig):
    label_descriptors: str = "label_descriptors.csv"
    labels: str = "lbl"


@register_task("audio_classification", dataclass=AudioClassificationConfig)
class AudioClassificationTask(AudioPretrainingTask):
    """ """

    cfg: AudioClassificationConfig

    def __init__(
        self,
        cfg: AudioClassificationConfig,
    ):
        super().__init__(cfg)

        self.state.add_factory("labels", self.load_labels)

    def load_labels(self):
        labels = {}
        path = os.path.join(self.cfg.data, self.cfg.label_descriptors)
        with open(path, "r") as ldf:
            for line in ldf:
                if line.strip() == "":
                    continue
                items = line.split(",")
                idx = items[0]
                lbl = items[1]
                assert lbl not in labels, lbl
                labels[lbl] = idx
        return labels

    @property
    def labels(self):
        return self.state.labels

    def load_dataset(
        self, split: str, task_cfg: AudioClassificationConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg

        data_path = self.cfg.data
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        labels = []
        with open(label_path, "r") as f:
            for i, line in enumerate(f):
                if i not in skipped_indices:
                    lbl_items = line.rstrip().split("\t")
                    labels.append([int(x) for x in lbl_items[2].split(",")])

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        self.datasets[split] = AddClassTargetDataset(
            self.datasets[split],
            labels,
            multi_class=True,
            add_to_input=True,
            num_classes=len(self.labels),
        )

    def calculate_stats(self, output, target):

        classes_num = target.shape[-1]
        stats = []

        # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
        # acc = sklearn_metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

        # Class-wise statistics
        for k in range(classes_num):
            # Average precision
            avg_precision = sklearn_metrics.average_precision_score(
                target[:, k], output[:, k], average=None
            )

            dict = {
                "AP": avg_precision,
            }

            # # AUC
            # try:
            #     auc = sklearn_metrics.roc_auc_score(target[:, k], output[:, k], average=None)
            # except:
            #     auc = 0
            #
            # # Precisions, recalls
            # (precisions, recalls, thresholds) = sklearn_metrics.precision_recall_curve(
            #     target[:, k], output[:, k]
            # )
            #
            # # FPR, TPR
            # (fpr, tpr, thresholds) = sklearn_metrics.roc_curve(target[:, k], output[:, k])
            #
            # save_every_steps = 1000  # Sample statistics to reduce size
            # dict = {
            #     "precisions": precisions[0::save_every_steps],
            #     "recalls": recalls[0::save_every_steps],
            #     "AP": avg_precision,
            #     "fpr": fpr[0::save_every_steps],
            #     "fnr": 1.0 - tpr[0::save_every_steps],
            #     "auc": auc,
            #     # note acc is not class-wise, this is just to keep consistent with other metrics
            #     "acc": acc,
            # }
            stats.append(dict)

        return stats

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if "_predictions" in logging_outputs[0]:
            metrics.log_concat_tensor(
                "_predictions",
                torch.cat([l["_predictions"].cpu() for l in logging_outputs], dim=0),
            )
            metrics.log_concat_tensor(
                "_targets",
                torch.cat([l["_targets"].cpu() for l in logging_outputs], dim=0),
            )

            def compute_stats(meters):
                if meters["_predictions"].tensor.shape[0] < 100:
                    return 0
                stats = self.calculate_stats(
                    meters["_predictions"].tensor, meters["_targets"].tensor
                )
                return np.nanmean([stat["AP"] for stat in stats])

            metrics.log_derived("mAP", compute_stats)
