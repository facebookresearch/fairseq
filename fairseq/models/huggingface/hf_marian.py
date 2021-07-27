# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    FairseqEncoderDecoderModel,
    TransformerDecoder, 
    FairseqEncoder, 
    FairseqIncrementalDecoder
) 

logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("hf_marian")
class HuggingFaceMarianNMT(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        
    @staticmethod
    def add_args(parser):
        """add model locations"""
        # fmt: off
        parser.add_argument('--model-path', type=int, metavar='N',
                            help='folder location for pretrained mdoel')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        try: 
            from transformers import (
                MarianMTEncoder, 
                MarianMTDecoder, 
                MarianConfig
            )
        except ImportError:
            raise ImportError(
                "\n\nPlease install huggingface/transformers with:"
                "\n\n  pip install transformers"
            )

        encoder = HuggingFaceMarianEncoder(args, task.dictionary)
        decoder = HuggingFaceMarianDecoder(args, task.dictionary)
        return cls(args, encoder, decoder)



class HuggingFaceMarianEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        config = MarianConfig.from_pretrained(args.model_path)
        self.model = MarianEncoder.from_pretrained(args.model_path)
        self.args = args
        self.dictionary = dictionary
        self.config = config

    def forward(self, src_tokens):
        

class HuggingFaceMarianDecoder(FairseqIncrementalDecoder):
