# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import glob
import numpy as np

from . import metric as metric_path
from . import predictor as predictor_path


class Evaluator(object):
    """
    perform evaluation on a single (downstream) task.
    make this both offline and online.
    TODO(huxu) saving evaluation results.
    """

    def __init__(self, config, eval_dataloader=None):
        if config.metric is None:
            raise ValueError("config.metric is", config.metric)
        metric_cls = getattr(metric_path, config.metric)
        self.metric = metric_cls(config)
        if config.predictor is None:
            raise ValueError("config.predictor is", config.predictor)
        predictor_cls = getattr(predictor_path, config.predictor)
        self.predictor = predictor_cls(config)
        self.eval_dataloader = eval_dataloader

    def __call__(self):
        try:
            print(self.predictor.pred_dir)
            for pred_file in glob.glob(
                    self.predictor.pred_dir + "/*_merged.npy"):
                outputs = np.load(pred_file)
                results = self.metric.compute_metrics(outputs)
                self.metric.print_computed_metrics(results)

            outputs = np.load(os.path.join(
                    self.predictor.pred_dir, "merged.npy"))
            results = self.metric.compute_metrics(outputs)
            return {"results": results, "metric": self.metric}
        except FileNotFoundError:
            print("\n[missing]", self.predictor.pred_dir)
            return {}

    def evaluate(self, model, eval_dataloader=None, output_file="merged"):
        if eval_dataloader is None:
            eval_dataloader = self.eval_dataloader
        outputs = self.predictor.predict_loop(
            model, eval_dataloader, output_file)
        results = self.metric.compute_metrics(**outputs)
        return results
