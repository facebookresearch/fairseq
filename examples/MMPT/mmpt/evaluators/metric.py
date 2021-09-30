# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import json


class Metric(object):
    def __init__(self, config, metric_names):
        self.metric_names = metric_names

    def best_metric(self, metric):
        return metric[self.metric_names[0]]

    def save_metrics(self, fn, metrics):
        with open(fn, "w") as fw:
            json.dump(fw, metrics)

    def print_computed_metrics(self, metrics):
        raise NotImplementedError


class RetrievalMetric(Metric):
    """
    this is modified from `howto100m/metrics.py`.
    History of changes:
    refactor as a class.
    add metric_key in __init__
    """

    def __init__(self, config, metric_names=["R1", "R5", "R10", "MR"]):
        super().__init__(config, metric_names)
        self.error = False  # TODO(huxu): add to config to print error.

    def compute_metrics(self, outputs, texts, **kwargs):
        x = outputs
        sx = np.sort(-x, axis=1)
        d = np.diag(-x)
        d = d[:, np.newaxis]
        ind = sx - d
        ind = np.where(ind == 0)
        ind = ind[1]
        metrics = {}
        metrics["R1"] = float(np.sum(ind == 0)) / len(ind)
        metrics["R5"] = float(np.sum(ind < 5)) / len(ind)
        metrics["R10"] = float(np.sum(ind < 10)) / len(ind)
        metrics["MR"] = np.median(ind) + 1

        max_idx = np.argmax(outputs, axis=1)
        if self.error:
            # print top-20 errors.
            error = []
            for ex_idx in range(20):
                error.append((texts[ex_idx], texts[max_idx[ex_idx]]))
            metrics["error"] = error
        return metrics

    def print_computed_metrics(self, metrics):
        r1 = metrics["R1"]
        r5 = metrics["R5"]
        r10 = metrics["R10"]
        mr = metrics["MR"]
        print(
            "R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}".format(
                r1, r5, r10, mr
            )
        )
        if "error" in metrics:
            print(metrics["error"])


class DiDeMoMetric(Metric):
    """
    History of changes:
    python 2.x to python 3.x.
    merge utils.py into eval to save one file.
    reference: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    Code to evaluate your results on the DiDeMo dataset.
    """
    def __init__(self, config, metric_names=["rank1", "rank5", "miou"]):
        super().__init__(config, metric_names)

    def compute_metrics(self, outputs, targets, **kwargs):
        assert len(outputs) == len(targets)
        rank1, rank5, miou = self._eval_predictions(outputs, targets)
        metrics = {
            "rank1": rank1,
            "rank5": rank5,
            "miou": miou
        }
        return metrics

    def print_computed_metrics(self, metrics):
        rank1 = metrics["rank1"]
        rank5 = metrics["rank5"]
        miou = metrics["miou"]
        # print("Average rank@1: %f" % rank1)
        # print("Average rank@5: %f" % rank5)
        # print("Average iou: %f" % miou)

        print(
            "Average rank@1: {:.4f} Average rank@5: {:.4f} Average iou: {:.4f}".format(
                rank1, rank5, miou
            )
        )

    def _iou(self, pred, gt):
        intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
        union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
        return float(intersection)/union

    def _rank(self, pred, gt):
        return pred.index(tuple(gt)) + 1

    def _eval_predictions(self, segments, data):
        '''
        Inputs:
        segments: For each item in the ground truth data, rank possible video segments given the description and video.
            In DiDeMo, there are 21 posible moments extracted for each video so the list of video segments will be of length 21.
            The first video segment should be the video segment that best corresponds to the text query.
            There are 4180 sentence in the validation data, so when evaluating a model on the val dataset,
            segments should be a list of lenght 4180, and each item in segments should be a list of length 21.
        data: ground truth data
        '''
        average_ranks = []
        average_iou = []
        for s, d in zip(segments, data):
            pred = s[0]
            ious = [self._iou(pred, t) for t in d['times']]
            average_iou.append(np.mean(np.sort(ious)[-3:]))
            ranks = [self._rank(s, t) for t in d['times'] if tuple(t) in s]  # if t in s] is added for s, e not in prediction.
            average_ranks.append(np.mean(np.sort(ranks)[:3]))
        rank1 = np.sum(np.array(average_ranks) <= 1)/float(len(average_ranks))
        rank5 = np.sum(np.array(average_ranks) <= 5)/float(len(average_ranks))
        miou = np.mean(average_iou)

        # print("Average rank@1: %f" % rank1)
        # print("Average rank@5: %f" % rank5)
        # print("Average iou: %f" % miou)
        return rank1, rank5, miou


class NLGMetric(Metric):
    def __init__(
        self,
        config,
        metric_names=[
            "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4",
            "METEOR", "ROUGE_L", "CIDEr"
        ]
    ):
        super().__init__(config, metric_names)
        # please install NLGEval from `https://github.com/Maluuba/nlg-eval`
        from nlgeval import NLGEval
        self.nlg = NLGEval()

    def compute_metrics(self, outputs, targets, **kwargs):
        return self.nlg.compute_metrics(
            hyp_list=outputs, ref_list=targets)

    def print_computed_metrics(self, metrics):
        Bleu_1 = metrics["Bleu_1"]
        Bleu_2 = metrics["Bleu_2"]
        Bleu_3 = metrics["Bleu_3"]
        Bleu_4 = metrics["Bleu_4"]
        METEOR = metrics["METEOR"]
        ROUGE_L = metrics["ROUGE_L"]
        CIDEr = metrics["CIDEr"]

        print(
            "Bleu_1: {:.4f} - Bleu_2: {:.4f} - Bleu_3: {:.4f} - Bleu_4: {:.4f} - METEOR: {:.4f} - ROUGE_L: {:.4f} - CIDEr: {:.4f}".format(
                Bleu_1, Bleu_2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr
            )
        )


class QAMetric(Metric):
    def __init__(
        self,
        config,
        metric_names=["acc"]
    ):
        super().__init__(config, metric_names)

    def compute_metrics(self, outputs, targets, **kwargs):
        from sklearn.metrics import accuracy_score
        return {"acc": accuracy_score(targets, outputs)}

    def print_computed_metrics(self, metrics):
        print("acc: {:.4f}".format(metrics["acc"]))


class COINActionSegmentationMetric(Metric):
    """
    COIN dataset listed 3 repos for Action Segmentation.
    Action Sets, NeuralNetwork-Viterbi, TCFPN-ISBA.
    The first and second are the same.
    https://github.com/alexanderrichard/action-sets/blob/master/eval.py

    Future reference for the third:
    `https://github.com/Zephyr-D/TCFPN-ISBA/blob/master/utils/metrics.py`
    """
    def __init__(self, config, metric_name=["frame_acc"]):
        super().__init__(config, metric_name)

    def compute_metrics(self, outputs, targets):
        n_frames = 0
        n_errors = 0
        n_errors = sum(outputs != targets)
        n_frames = len(targets)
        return {"frame_acc": 1.0 - float(n_errors) / n_frames}

    def print_computed_metrics(self, metrics):
        fa = metrics["frame_acc"]
        print("frame accuracy:", fa)


class CrossTaskMetric(Metric):
    def __init__(self, config, metric_names=["recall"]):
        super().__init__(config, metric_names)

    def compute_metrics(self, outputs, targets, **kwargs):
        """refactored from line 166:
        https://github.com/DmZhukov/CrossTask/blob/master/train.py"""

        recalls = self._get_recalls(Y_true=targets, Y_pred=outputs)
        results = {}
        for task, rec in recalls.items():
            results[str(task)] = rec

        avg_recall = np.mean(list(recalls.values()))
        results["recall"] = avg_recall
        return results

    def print_computed_metrics(self, metrics):
        print('Recall: {0:0.3f}'.format(metrics["recall"]))
        for task in metrics:
            if task != "recall":
                print('Task {0}. Recall = {1:0.3f}'.format(
                    task, metrics[task]))

    def _get_recalls(self, Y_true, Y_pred):
        """refactored from
        https://github.com/DmZhukov/CrossTask/blob/master/train.py"""

        step_match = {task: 0 for task in Y_true.keys()}
        step_total = {task: 0 for task in Y_true.keys()}
        for task, ys_true in Y_true.items():
            ys_pred = Y_pred[task]
            for vid in set(ys_pred.keys()).intersection(set(ys_true.keys())):
                y_true = ys_true[vid]
                y_pred = ys_pred[vid]
                step_total[task] += (y_true.sum(axis=0) > 0).sum()
                step_match[task] += (y_true*y_pred).sum()
        recalls = {
            task: step_match[task] / n for task, n in step_total.items()}
        return recalls


class ActionRecognitionMetric(Metric):
    def __init__(
        self,
        config,
        metric_names=["acc", "acc_splits", "r1_splits", "r5_splits", "r10_splits"]
    ):
        super().__init__(config, metric_names)

    def compute_metrics(self, outputs, targets, splits, **kwargs):
        all_video_embd = outputs
        labels = targets
        split1, split2, split3 = splits
        accs = []
        r1s = []
        r5s = []
        r10s = []
        for split in range(3):
            if split == 0:
                s = split1
            elif split == 1:
                s = split2
            else:
                s = split3

            X_pred = all_video_embd[np.where(s == 2)[0]]
            label_test = labels[np.where(s == 2)[0]]
            logits = X_pred
            X_pred = np.argmax(X_pred, axis=1)
            acc = np.sum(X_pred == label_test) / float(len(X_pred))
            accs.append(acc)
            # compute recall.
            sorted_pred = (-logits).argsort(axis=-1)
            label_test_sp = label_test.reshape(-1, 1)

            r1 = np.mean((sorted_pred[:, :1] == label_test_sp).sum(axis=1), axis=0)
            r5 = np.mean((sorted_pred[:, :5] == label_test_sp).sum(axis=1), axis=0)
            r10 = np.mean((sorted_pred[:, :10] == label_test_sp).sum(axis=1), axis=0)
            r1s.append(r1)
            r5s.append(r5)
            r10s.append(r10)

        return {"acc": accs[0], "acc_splits": accs, "r1_splits": r1s, "r5_splits": r5s, "r10_splits": r10s}

    def print_computed_metrics(self, metrics):
        for split, acc in enumerate(metrics["acc_splits"]):
            print("Top 1 accuracy on split {}: {}; r1 {}; r5 {}; r10 {}".format(
                split + 1, acc,
                metrics["r1_splits"][split],
                metrics["r5_splits"][split],
                metrics["r10_splits"][split],
                )
            )
