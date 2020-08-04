# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import utils
from . import bleu, build_scoring


def build_scorer(args, tgt_dict):
    if args.sacrebleu:
        utils.deprecation_warning(
            "--sacrebleu is deprecated. Please use --scoring sacrebleu instead."
        )
        args.scoring = "sacrebleu"

    if args.scoring == "bleu":
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    else:
        return build_scoring(args)

    return scorer
