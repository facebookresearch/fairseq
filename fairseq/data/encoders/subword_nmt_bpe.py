# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import file_utils
from fairseq.data.encoders import register_bpe


@register_bpe('subword_nmt')
class SubwordNMTBPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--bpe-codes', type=str,
                            help='path to subword NMT BPE')
        parser.add_argument('--bpe-separator', default='@@',
                            help='BPE separator')
        # fmt: on

    @classmethod
    def from_args(cls, args):
        return cls(args.bpe_codes, args.bpe_separator)

    @classmethod
    def build_bpe(cls, args):
        return cls.from_args(args)

    def __init__(self, bpe_codes, bpe_separator):
        if bpe_codes is None:
            raise ValueError('--bpe-codes is required for --bpe=subword_nmt')
        codes = file_utils.cached_path(bpe_codes)
        try:
            from subword_nmt import apply_bpe
            bpe_parser = apply_bpe.create_parser()
            bpe_args = bpe_parser.parse_args([
                '--codes', codes,
                '--separator', bpe_separator,
            ])
            self.bpe = apply_bpe.BPE(
                bpe_args.codes,
                bpe_args.merges,
                bpe_args.separator,
                None,
                bpe_args.glossaries,
            )
            self.bpe_symbol = bpe_args.separator + ' '
        except ImportError:
            raise ImportError('Please install subword_nmt with: pip install subword-nmt')

    def encode(self, x: str) -> str:
        return self.bpe.process_line(x)

    def decode(self, x: str) -> str:
        return (x + ' ').replace(self.bpe_symbol, '').rstrip()
