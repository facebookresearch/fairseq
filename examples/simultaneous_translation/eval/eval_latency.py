# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from examples.simultaneous_translation.utils.latency import LatencyInference
import argparse
import torch
import json


LATENCY_METRICS = [
    'differentiable_average_lagging',
    'average_lagging',
    'average_proportion',
]


class LatencyScorer():
    def __init__(self, start_from_zero=True):
        self.recorder = []
        self.scores = {}
        self.scorer = LatencyInference()
        self.start_from_zero = start_from_zero

    def update_reorder(self, list_of_dict):
        self.recorder = []
        for info in list_of_dict:
            delays = [
                int(x) - int(not self.start_from_zero)
                for x in info["delays"]
            ]
            delays = torch.LongTensor(delays).unsqueeze(0)
            src_len = torch.LongTensor([info["src_len"]]).unsqueeze(0)

            self.recorder.append(self.scorer(delays, src_len))

    def cal_latency(self):
        self.scores = {}
        for metric in LATENCY_METRICS:
            self.scores[metric] = sum(
                [x[metric][0, 0].item() for x in self.recorder]
            ) / len(self.recorder)
        return self.scores

    @classmethod
    def score(cls, list_of_dict, start_from_zero=True):
        scorer_to_return = cls(start_from_zero)
        scorer_to_return.update_reorder(list_of_dict)
        scorer_to_return.cal_latency()
        return scorer_to_return.scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--start-from-zero", action="store_true")
    args = parser.parse_args()

    scorer = LatencyInference()
    recorder = []
    with open(args.input, 'r') as f:
        for line in f:
            info = json.loads(line)

            delays = [int(x) - int(not args.start_from_zero) for x in info["delays"]]

            delays = torch.LongTensor(delays).unsqueeze(0)

            src_len = torch.LongTensor([info["src_len"]]).unsqueeze(0)

            recorder.append(scorer(delays, src_len))

    average_results = {}

    for metric in LATENCY_METRICS:
        average_results[metric] = sum(
            [x[metric][0, 0].item() for x in recorder]
        ) / len(recorder)
        print(f"{metric}: {average_results[metric]}")
