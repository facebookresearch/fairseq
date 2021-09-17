
import argparse
import os
import numpy as np
import itertools
import regex

# U+2581
spm_underscore = "▁"
spm_underscore2 = u"\u2581"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.input)
    assert os.path.exists(args.vocab)
    # assert os.path.exists(args.output)

    with open(args.input, 'r', encoding='utf-8') as f:
        toks = list(itertools.chain.from_iterable(x.rstrip().split(" ") for x in f.read().splitlines()))
    
    with open(args.vocab, 'r', encoding='utf-8') as f:
        vocab = [x.rstrip().rsplit(" ", 1) for x in f.readlines()]
        vocab = [(x[0], int(x[1])) for x in vocab]
    
    xdict = {}
    for t in toks:
        xdict[t] = xdict.get(t, 0) + 1
    
    nvocab = [(x[0], xdict.get(x[0], 0)) for x in vocab]
    with open(args.output, 'w', encoding='utf-8') as f:
        f.writelines([f'{x[0]} {x[1]}\n' for x in nvocab])

