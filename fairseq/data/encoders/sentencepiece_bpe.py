# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import file_utils
from fairseq.data.encoders import register_bpe


@register_bpe('sentencepiece')
class SentencepieceBPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--sentencepiece-vocab', type=str,
                            help='path to sentencepiece vocab')
        # fmt: on

    def __init__(self, args):
        vocab = file_utils.cached_path(args.sentencepiece_vocab)
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(vocab)
        except ImportError:
            raise ImportError('Please install sentencepiece with: pip install sentencepiece')

    def encode(self, x: str) -> str:
        return ' '.join(self.sp.EncodeAsPieces(x))

    def decode(self, x: str) -> str:
        return x.replace(' ', '').replace('\u2581', ' ').strip()
