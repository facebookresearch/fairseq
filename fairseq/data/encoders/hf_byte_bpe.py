# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data.encoders import register_bpe


@register_bpe('hf_byte_bpe')
class HuggingFaceByteLevelBPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--bpe-merges', help='path to merges.txt')
        parser.add_argument('--bpe-vocab', help='path to vocab.json')
        parser.add_argument('--bpe-add-prefix-space', action='store_true',
                            help='add prefix space before encoding')
        # fmt: on

    def __init__(self, args):
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                'Please install huggingface/tokenizers with: '
                'pip install tokenizers'
            )

        self.bpe = ByteLevelBPETokenizer(
            args.bpe_vocab,
            args.bpe_merges,
            add_prefix_space=getattr(args, 'bpe_add_prefix_space', False),
        )

    def encode(self, x: str) -> str:
        return ' '.join(map(str, self.bpe.encode(x).ids))

    def decode(self, x: str) -> str:
        return self.bpe.decode([
            int(tok) if tok not in {'<unk>', '<mask>'} else tok
            for tok in x.split()
        ])

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(' ')
