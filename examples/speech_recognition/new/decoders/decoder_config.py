# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional

from fairseq.dataclass.configs import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from omegaconf import MISSING


DECODER_CHOICES = ChoiceEnum(["viterbi", "kenlm", "fairseqlm"])


@dataclass
class DecoderConfig(FairseqDataclass):
    type: DECODER_CHOICES = field(
        default="viterbi",
        metadata={"help": "The type of decoder to use"},
    )


@dataclass
class FlashlightDecoderConfig(FairseqDataclass):
    nbest: int = field(
        default=1,
        metadata={"help": "Number of decodings to return"},
    )
    unitlm: bool = field(
        default=False,
        metadata={"help": "If set, use unit language model"},
    )
    lmpath: str = field(
        default=MISSING,
        metadata={"help": "Language model for KenLM decoder"},
    )
    lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "Lexicon for Flashlight decoder"},
    )
    beam: int = field(
        default=50,
        metadata={"help": "Number of beams to use for decoding"},
    )
    beamthreshold: float = field(
        default=50.0,
        metadata={"help": "Threshold for beam search decoding"},
    )
    beamsizetoken: Optional[int] = field(
        default=None, metadata={"help": "Beam size to use"}
    )
    wordscore: float = field(
        default=-1,
        metadata={"help": "Word score for KenLM decoder"},
    )
    unkweight: float = field(
        default=-math.inf,
        metadata={"help": "Unknown weight for KenLM decoder"},
    )
    silweight: float = field(
        default=0,
        metadata={"help": "Silence weight for KenLM decoder"},
    )
    lmweight: float = field(
        default=2,
        metadata={"help": "Weight for LM while interpolating score"},
    )
