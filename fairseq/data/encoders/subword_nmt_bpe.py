# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

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

    def __init__(self, args):
        if args.bpe_codes is None:
            raise ValueError('--bpe-codes is required for --bpe=subword_nmt')
        codes = file_utils.cached_path(args.bpe_codes)
        try:
            from subword_nmt import apply_bpe
            bpe_parser = apply_bpe.create_parser()
            bpe_args = bpe_parser.parse_args([
                '--codes', codes,
                '--separator', args.bpe_separator,
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
