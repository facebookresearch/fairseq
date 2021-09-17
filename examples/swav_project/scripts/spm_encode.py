
import os
from turtle import width
import numpy as np
import argparse
import sentencepiece as spm

import functools
"""

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--old_params', default=False, action='store_true')
    args = parser.parse_args()
    s = spm.SentencePieceProcessor(model_file=args.model)
    if args.old_params:
        encode = lambda x: s.encode(x, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
    else:
        encode = lambda x: s.encode(x, out_type=str)
    with open(args.input, 'r', encoding='utf-8') as fi:
        with open(args.out, 'w', encoding='utf-8') as fo:
            lines = fi.readlines()
            for i, l in enumerate(lines):
                l = l.strip()
                if len(l) > 0:
                    # out_l = s.encode(l, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
                    out_l = encode(l)
                    out_l = " ".join(out_l)
                    fo.write(f'{out_l}\n')
                if i % 100000 == 0:
                    print(i)
    print(f'Wrote: {args.out}')






