# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import os

from fairseq import registry


_build_scoring, register_scoring, SCORING_REGISTRY, _ = registry.setup_registry(
    "--scoring", default="bleu"
)


def build_scorer(args, tgt_dict):
    from fairseq import utils

    if args.sacrebleu:
        utils.deprecation_warning(
            "--sacrebleu is deprecated. Please use --scoring sacrebleu instead."
        )
        args.scoring = "sacrebleu"
    if args.scoring == "bleu":
        from fairseq.scoring import bleu
        return bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    else:
        return _build_scoring(args)


# automatically import any Python files in the current directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("fairseq.scoring." + module)
