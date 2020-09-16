# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import os
from abc import ABC, abstractmethod

from fairseq import registry
from fairseq import utils


class BaseScorer(ABC):
    def __init__(self, args):
        self.args = args
        self.ref = []
        self.pred = []

    @staticmethod
    def add_args(parser):
        pass

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.pred.append(pred)

    @abstractmethod
    def score(self) -> float:
        pass

    @abstractmethod
    def result_string(self) -> str:
        pass


_build_scorer, register_scorer, SCORER_REGISTRY = registry.setup_registry(
    "--scoring", default="bleu"
)


# automatically import any Python files in the current directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("fairseq.scorers." + module)


def build_scorer(args, tgt_dict):
    if args.scoring == "bleu":
        from .bleu import Scorer as BLEUScorer
        return BLEUScorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())

    if args.sacrebleu:
        utils.deprecation_warning(
            "--sacrebleu is deprecated. Please use --scoring sacrebleu instead."
        )

    return _build_scorer(args)
