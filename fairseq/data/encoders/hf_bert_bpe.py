# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data.encoders import register_bpe


@register_bpe('bert')
class BertBPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--bpe-cased', action='store_true',
                            help='set for cased BPE',
                            default=False)
        parser.add_argument('--bpe-vocab-file', type=str,
                            help='bpe vocab file.')
        # fmt: on

    def __init__(self, args):
        try:
            from pytorch_transformers import BertTokenizer
            from pytorch_transformers.tokenization_utils import clean_up_tokenization
        except ImportError:
            raise ImportError(
                'Please install 1.0.0 version of pytorch_transformers'
                'with: pip install pytorch-transformers'
            )

        if 'bpe_vocab_file' in args:
            self.bert_tokenizer = BertTokenizer(
                args.bpe_vocab_file,
                do_lower_case=not args.bpe_cased
            )
        else:
            vocab_file_name = 'bert-base-cased' if args.bpe_cased else 'bert-base-uncased'
            self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file_name)
            self.clean_up_tokenization = clean_up_tokenization

    def encode(self, x: str) -> str:
        return ' '.join(self.bert_tokenizer.tokenize(x))

    def decode(self, x: str) -> str:
        return self.clean_up_tokenization(
            self.bert_tokenizer.convert_tokens_to_string(x.split(' '))
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return not x.startswith('##')
