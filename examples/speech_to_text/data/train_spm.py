#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
from pathlib import Path
import os
import json
from typing import List, Optional
import re

import yaml
import sentencepiece as sp
from tqdm import tqdm
import argparse
from multiprocessing import cpu_count
from constant import UNK, BOS, EOS, PAD


def gen_dict_txt(in_path: str, out_dir_path: str):
    out_path = Path(out_dir_path) / 'dict.txt'
    with open(in_path) as f_in, open(out_path, 'w') as f_out:
        for l_in in f_in:
            s, c = l_in.strip().split()
            if s in {UNK, BOS, EOS, PAD}:
                continue
            f_out.write(f'{s} 1\n')

def train_spm_vocab(
        input_path: str, out_root: str, out_suffix='', vocab_size=1000,
        model_type='bpe', max_input_sentence=10000000,
):
    if not Path(out_root).is_dir():
        os.makedirs(out_root)

    model_prefix = Path(out_root) / f'spm_{model_type}_{vocab_size}{out_suffix}'
    #don't change the value of unk_id, bos_id, eos_id and pad_id, so they will have
    #the same default values as dictionary at fairseq/data/dictionary.py 
    arguments = [
        f'--input={input_path}',
        f'--model_prefix={model_prefix}',
        f'--model_type={model_type}',
        f'--vocab_size={vocab_size}',
        '--character_coverage=1.0',
        '--shuffle_input_sentence',
        f'--num_threads={cpu_count()}',
        f'--input_sentence_size={max_input_sentence}',
        '--unk_id=3',
        '--bos_id=0',
        '--eos_id=2',
        '--pad_id=1'
    ]
    sp.SentencePieceTrainer.Train(' '.join(arguments))
    return model_prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--lang", type=str, default="")
    args = parser.parse_args()

    output_name = f"{args.model_type}-{args.lang}-{args.vocab_size}" if args.lang !="" else f"{args.model_type}-{args.vocab_size}"
    output_path = os.path.join(args.out_path, output_name)
    if not os.path.exists(output_path ):
        os.mkdir(output_path)

    model_prefix = train_spm_vocab(
        input_path=args.data_path ,
        out_root=os.path.join(args.out_path, output_name, 'vocab'),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )

    try:
        os.unlink(os.path.join(output_path, 'spm.model'))
    except:
        pass

    os.symlink(
        os.path.join(
            'vocab',
            os.path.basename(model_prefix.__str__())+ '.model'
        ),
        os.path.join(output_path, 'spm.model')
    )

    gen_dict_txt(
        in_path=model_prefix.__str__()+ '.vocab',
        out_dir_path=os.path.join(args.out_path, output_name)
    )
