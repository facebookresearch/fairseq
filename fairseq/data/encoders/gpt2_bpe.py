# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import file_utils
from fairseq.data.encoders import register_bpe

from .gpt2_bpe_utils import get_encoder


DEFAULT_ENCODER_JSON = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'


@register_bpe('gpt2')
class GPT2BPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--gpt2-encoder-json', type=str,
                            default=DEFAULT_ENCODER_JSON,
                            help='path to encoder.json')
        parser.add_argument('--gpt2-vocab-bpe', type=str,
                            default=DEFAULT_VOCAB_BPE,
                            help='path to vocab.bpe')
        # fmt: on

    def __init__(self, args):
        encoder_json = file_utils.cached_path(
            getattr(args, 'gpt2_encoder_json', DEFAULT_ENCODER_JSON)
        )
        vocab_bpe = file_utils.cached_path(
            getattr(args, 'gpt2_vocab_bpe', DEFAULT_VOCAB_BPE)
        )
        self.bpe = get_encoder(encoder_json, vocab_bpe)

    def encode(self, x: str) -> str:
        return ' '.join(map(str, self.bpe.encode(x)))

    def decode(self, x: str) -> str:
        return self.bpe.decode(map(int, x.split()))

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(' ')
