# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import file_utils
from fairseq.data.encoders import register_bpe


@register_bpe("sentencepiece")
class SentencepieceBPE(object):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--sentencepiece-model', type=str,
                            help='path to sentencepiece model')
        # fmt: on

    def __init__(self, args):
        sentencepiece_model = file_utils.cached_path(args.sentencepiece_model)
        try:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sentencepiece_model)
        except ImportError:
            raise ImportError(
                "Please install sentencepiece with: pip install sentencepiece"
            )

    def encode(self, x: str) -> str:
        return " ".join(self.sp.EncodeAsPieces(x))

    def decode(self, x: str) -> str:
        return x.replace(" ", "").replace("\u2581", " ").strip()

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>"]:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith("\u2581")
