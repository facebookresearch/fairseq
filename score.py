#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import argparse
import os
import sys

from fairseq import bleu, tokenizer
from fairseq.data import dictionary


def main():
    parser = argparse.ArgumentParser(description='Command-line script for BLEU scoring.')
    parser.add_argument('-s', '--sys', default='-', help='system output')
    parser.add_argument('-r', '--ref', required=True, help='references')
    parser.add_argument('-o', '--order', default=4, metavar='N',
                        type=int, help='consider ngrams up to this order')
    parser.add_argument('--ignore-case', action='store_true',
                        help='case-insensitive scoring')
    parser.add_argument('--tokenizer_name', metavar='N', default='default', choices=['default', 'nltk', 'sacremoses'],
                        help="Which tokenizer to use. Choices are default, nltk, sacremoses. default tokenizes by splitting on white space. nltk uses "
                             "nltk's word_tokenize which better takes into account punctuation. As an example "
                             "'Hello, how's your day today?' would be tokenized as "
                             "['Hello,' , 'how's', 'your', 'day', 'today?'] when using the default, but would instead be tokenized as "
                             "['Hello', ',', 'how', ''s', 'your', 'day', 'today', '?'] when using nltk. The sacremoses tokenizer is from this package, "
                             "https://github.com/alvations/sacremoses.")
    args = parser.parse_args()
    print(args)

    assert args.sys == '-' or os.path.exists(args.sys), \
        "System output file {} does not exist".format(args.sys)
    assert os.path.exists(args.ref), \
        "Reference file {} does not exist".format(args.ref)

    dict = dictionary.Dictionary()
    tokenizer_tool = tokenizer.build_tokenizer(args)

    def readlines(fd):
        for line in fd.readlines():
            if args.ignore_case:
                yield line.lower()
            yield line

    def score(fdsys):
        with open(args.ref) as fdref:
            scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
            for sys_tok, ref_tok in zip(readlines(fdsys), readlines(fdref)):
                sys_tok = tokenizer_tool.tokenize(sys_tok, dict)
                ref_tok = tokenizer_tool.tokenize(ref_tok, dict)
                scorer.add(ref_tok, sys_tok)
            print(scorer.result_string(args.order))

    if args.sys == '-':
        score(sys.stdin)
    else:
        with open(args.sys, 'r') as f:
            score(f)


if __name__ == '__main__':
    main()
