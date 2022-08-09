# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import collections
import argparse
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
import sentencepiece as spm


def preprocess(spm_model_path, train_path, valid_path, test_path, dest_dir, remove_empty=False, output_format='piece', workers=20):
    with tempfile.TemporaryDirectory() as tmp:
        # Tokenize with SentencePiece
        for split, path in ('train', train_path), ('valid', valid_path), ('test', test_path):
            if path is None:
                continue
            if path == '-':
                path = sys.stdin.fileno()
            with open(path, encoding='utf-8', errors='surrogateescape') as fin:
                with open(f'{tmp}/{split}', mode='w', encoding='utf-8', errors='surrogateescape') as fout:
                    encoder = MultiprocessingEncoder(model=spm_model_path, remove_empty=remove_empty, output_format=output_format)
                    pool = Pool(workers, initializer=encoder.initializer)
                    encoded_lines = pool.imap(encoder.encode, fin, 10000)
                    for i, line in enumerate(encoded_lines, start=1):
                        if line is not None:
                            print(line, file=fout)
                        if i % 10000 == 0:
                            print("tokenized {} lines".format(i), file=sys.stderr)

        # Generate dictionary
        sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        if output_format == 'piece':
            vocab = [sp.id_to_piece(i) for i in range(3, sp.vocab_size())]
        else:
            vocab = map(str, range(sp.vocab_size()))
        with open(f'{tmp}/dict.txt', mode='w', encoding='utf-8', errors='surrogateescape') as f:
            for word in vocab:
                print(word, 1, file=f)

        # Binarize
        command = [
            'python3', '-m', 'fairseq_cli.preprocess',
            '--only-source',
            '--thresholdsrc', '0',
            '--destdir', dest_dir,
            '--srcdict', f'{tmp}/dict.txt',
            '--workers', '20',
        ]
        for split, path in ('train', train_path), ('valid', valid_path), ('test', test_path):
            if path is not None:
                command += [f'--{split}pref', f'{tmp}/{split}']
        subprocess.run(command)
        
        # Copy SentencePiece model
        shutil.copyfile(spm_model_path, f'{dest_dir}/sentencepiece.bpe.model')


class MultiprocessingEncoder(object):
    def __init__(self, model, remove_empty, output_format):
        self.model = model
        self.remove_empty = remove_empty
        self.output_format = output_format

    def initializer(self):
        global sp
        sp = spm.SentencePieceProcessor(model_file=self.model)

    def encode(self, line):
        global sp
        line = line.strip()
        if len(line) == 0 and self.remove_empty:
            return None

        if self.output_format == 'piece':
            return ' '.join(sp.encode_as_pieces(line))
        else:
            return ' '.join(map(str, sp.encode(line)))


def write_lines(lines, path):
    with open(path, mode='x', encoding='utf-8') as f:
        for line in lines:
            print(line, file=f)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f.read().splitlines()]


def read_nli(path, langs=None):
    data = read_jsonl(path)

    if langs is not None:
        data = [sample for sample in data if sample.get('language') in langs]

    lang2count = collections.defaultdict(int)
    for sample in data:
        lang2count[sample.get('language')] += 1

    if langs:
        assert set(lang2count.keys()) == set(langs)

    nlangs = len(lang2count)
    assert nlangs > 0
    lens = list(lang2count.values())
    assert all([lens[0] == length for length in lens])

    print(f'Loaded {lens[0]} samples in {nlangs} languages from {path}', file=sys.stderr)
    return data


def main():
    parser = argparse.ArgumentParser(description='Tokenize and binarize NLI data')
    parser.add_argument('--sentencepiece-model', required=True)
    parser.add_argument('--train', required=True, help='Training data in jsonl format')
    parser.add_argument('--valid', required=True, help='Validation data in jsonl format')
    parser.add_argument('--destdir', required=True)

    args = parser.parse_args()

    os.makedirs(args.destdir + '/raw',)
    os.makedirs(args.destdir + '/bin', )

    # Extract input/labels
    for split, path in ('train', args.train), ('valid', args.valid):
        data = read_nli(path, langs=None)
        original_size = len(data)
        data = [sample for sample in data if sample['gold_label'] != '-']
        assert all(sample['gold_label'] in ('contradiction', 'entailment', 'neutral') for sample in data)
        filtered_size = len(data)
        if filtered_size != original_size:
            print(f'Filtered {filtered_size}/{original_size} samples from {path}', file=sys.stderr)
        for name, field in ('input0', 'sentence1'), ('input1', 'sentence2'), ('label', 'gold_label'):
            write_lines([sample[field] for sample in data], f'{args.destdir}/raw/{split}.{name}.txt')

    # Tokenize and binarize input
    for field in 'input0', 'input1':
        preprocess(
            spm_model_path=args.sentencepiece_model,
            train_path=f'{args.destdir}/raw/train.{field}.txt',
            valid_path=f'{args.destdir}/raw/valid.{field}.txt',
            test_path=None,
            dest_dir=f'{args.destdir}/bin/{field}',
            workers=20,
        )
    
    # Binarize labels
    subprocess.run([
        'python3', '-m', 'fairseq_cli.preprocess',
        '--trainpref', f'{args.destdir}/raw/train.label.txt',
        '--validpref', f'{args.destdir}/raw/valid.label.txt',
        '--only-source',
        '--thresholdsrc', '0',
        '--destdir', f'{args.destdir}/bin/label',
        '--workers', '20',
    ])


if __name__ == '__main__':
    main()
