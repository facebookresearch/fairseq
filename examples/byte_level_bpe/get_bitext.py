# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os.path as op
import argparse
import os
from multiprocessing import cpu_count
from collections import namedtuple
from typing import Optional, List

import sentencepiece as sp

from fairseq.data.encoders.moses_tokenizer import MosesTokenizer
from fairseq.data.encoders.byte_utils import byte_encode
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
from fairseq.data.encoders.characters import Characters
from fairseq.data.encoders.byte_bpe import ByteBPE
from fairseq.data.encoders.bytes import Bytes


SPLITS = ['train', 'valid', 'test']


def _convert_xml(in_path: str, out_path: str):
    with open(in_path) as f, open(out_path, 'w') as f_o:
        for s in f:
            ss = s.strip()
            if not ss.startswith('<seg'):
                continue
            ss = ss.replace('</seg>', '').split('">')
            assert len(ss) == 2
            f_o.write(ss[1].strip() + '\n')


def _convert_train(in_path: str, out_path: str):
    with open(in_path) as f, open(out_path, 'w') as f_o:
        for s in f:
            ss = s.strip()
            if ss.startswith('<'):
                continue
            f_o.write(ss.strip() + '\n')


def _get_bytes(in_path: str, out_path: str):
    with open(in_path) as f, open(out_path, 'w') as f_o:
        for s in f:
            f_o.write(Bytes.encode(s.strip()) + '\n')


def _get_chars(in_path: str, out_path: str):
    with open(in_path) as f, open(out_path, 'w') as f_o:
        for s in f:
            f_o.write(Characters.encode(s.strip()) + '\n')


def pretokenize(in_path: str, out_path: str, src: str, tgt: str):
    Args = namedtuple('Args', ['moses_source_lang', 'moses_target_lang',
                               'moses_no_dash_splits', 'moses_no_escape'])
    args = Args(moses_source_lang=src, moses_target_lang=tgt,
                moses_no_dash_splits=False, moses_no_escape=False)
    pretokenizer = MosesTokenizer(args)
    with open(in_path) as f, open(out_path, 'w') as f_o:
        for s in f:
            f_o.write(pretokenizer.encode(s.strip()) + '\n')


def _convert_to_bchar(in_path_prefix: str, src: str, tgt: str, out_path: str):
    with open(out_path, 'w') as f_o:
        for lang in [src, tgt]:
            with open(f'{in_path_prefix}.{lang}') as f:
                for s in f:
                    f_o.write(byte_encode(s.strip()) + '\n')


def _get_bpe(in_path: str, model_prefix: str, vocab_size: int):
    arguments = [
        f'--input={in_path}', f'--model_prefix={model_prefix}',
        f'--model_type=bpe', f'--vocab_size={vocab_size}',
        '--character_coverage=1.0', '--normalization_rule_name=identity',
        f'--num_threads={cpu_count()}'
    ]
    sp.SentencePieceTrainer.Train(' '.join(arguments))


def _apply_bbpe(model_path: str, in_path: str, out_path: str):
    Args = namedtuple('Args', ['sentencepiece_model_path'])
    args = Args(sentencepiece_model_path=model_path)
    tokenizer = ByteBPE(args)
    with open(in_path) as f, open(out_path, 'w') as f_o:
        for s in f:
            f_o.write(tokenizer.encode(s.strip()) + '\n')


def _apply_bpe(model_path: str, in_path: str, out_path: str):
    Args = namedtuple('Args', ['sentencepiece_vocab'])
    args = Args(sentencepiece_vocab=model_path)
    tokenizer = SentencepieceBPE(args)
    with open(in_path) as f, open(out_path, 'w') as f_o:
        for s in f:
            f_o.write(tokenizer.encode(s.strip()) + '\n')


def _concat_files(in_paths: List[str], out_path: str):
    with open(out_path, 'w') as f_o:
        for p in in_paths:
            with open(p) as f:
                for r in f:
                    f_o.write(r)


def preprocess_iwslt17(root: str, src: str, tgt: str, bpe_size: Optional[int],
                       need_chars: bool, bbpe_size: Optional[int],
                       need_bytes: bool):
    # extract bitext
    in_root = op.join(root, f'{src}-{tgt}')
    for lang in [src, tgt]:
        _convert_train(
            op.join(in_root, f'train.tags.{src}-{tgt}.{lang}'),
            op.join(root, f'train.{lang}')
        )
        _convert_xml(
            op.join(in_root, f'IWSLT17.TED.dev2010.{src}-{tgt}.{lang}.xml'),
            op.join(root, f'valid.{lang}')
        )
        _convert_xml(
            op.join(in_root, f'IWSLT17.TED.tst2015.{src}-{tgt}.{lang}.xml'),
            op.join(root, f'test.{lang}')
        )
    # pre-tokenize
    for lang in [src, tgt]:
        for split in SPLITS:
            pretokenize(op.join(root, f'{split}.{lang}'),
                        op.join(root, f'{split}.moses.{lang}'), src, tgt)
    # tokenize with BPE vocabulary
    if bpe_size is not None:
        # learn vocabulary
        concated_train_path = op.join(root, 'train.all')
        _concat_files(
            [op.join(root, 'train.moses.fr'), op.join(root, 'train.moses.en')],
            concated_train_path
        )
        bpe_model_prefix = op.join(root, f'spm_bpe{bpe_size}')
        _get_bpe(concated_train_path, bpe_model_prefix, bpe_size)
        os.remove(concated_train_path)
        # apply
        for lang in [src, tgt]:
            for split in SPLITS:
                _apply_bpe(
                    bpe_model_prefix + '.model',
                    op.join(root, f'{split}.moses.{lang}'),
                    op.join(root, f'{split}.moses.bpe{bpe_size}.{lang}')
                )
    # tokenize with bytes vocabulary
    if need_bytes:
        for lang in [src, tgt]:
            for split in SPLITS:
                _get_bytes(op.join(root, f'{split}.moses.{lang}'),
                           op.join(root, f'{split}.moses.bytes.{lang}'))
    # tokenize with characters vocabulary
    if need_chars:
        for lang in [src, tgt]:
            for split in SPLITS:
                _get_chars(op.join(root, f'{split}.moses.{lang}'),
                           op.join(root, f'{split}.moses.chars.{lang}'))
    # tokenize with byte-level BPE vocabulary
    if bbpe_size is not None:
        # learn vocabulary
        bchar_path = op.join(root, 'train.bchar')
        _convert_to_bchar(op.join(root, 'train.moses'), src, tgt, bchar_path)
        bbpe_model_prefix = op.join(root, f'spm_bbpe{bbpe_size}')
        _get_bpe(bchar_path, bbpe_model_prefix, bbpe_size)
        os.remove(bchar_path)
        # apply
        for lang in [src, tgt]:
            for split in SPLITS:
                _apply_bbpe(
                    bbpe_model_prefix + '.model',
                    op.join(root, f'{split}.moses.{lang}'),
                    op.join(root, f'{split}.moses.bbpe{bbpe_size}.{lang}')
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--bpe-vocab', default=None, type=int,
                        help='Generate tokenized bitext with BPE of size K.'
                             'Default to None (disabled).')
    parser.add_argument('--bbpe-vocab', default=None, type=int,
                        help='Generate tokenized bitext with BBPE of size K.'
                             'Default to None (disabled).')
    parser.add_argument('--byte-vocab', action='store_true',
                        help='Generate tokenized bitext with bytes vocabulary')
    parser.add_argument('--char-vocab', action='store_true',
                        help='Generate tokenized bitext with chars vocabulary')
    args = parser.parse_args()

    preprocess_iwslt17(args.root, 'fr', 'en', args.bpe_vocab, args.char_vocab,
                       args.bbpe_vocab, args.byte_vocab)


if __name__ == '__main__':
    main()
