#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Scoring script for computing pairwise BLEU and oracle BLEU over a set of
candidate hypotheses.

See `"Mixture Models for Diverse Machine Translation: Tricks of the Trade"
(Shen et al., 2019) <https://arxiv.org/abs/1902.07816>`_.
"""

import argparse
import sys
import numpy as np
import random

from fairseq import bleu, tokenizer
from fairseq.data import dictionary

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--sys', nargs='*', default='', metavar='FILE',
                    help='path to system output')
parser.add_argument('--ref', default='', metavar='FILE',
                    help='path to references')
parser.add_argument('--output', default='', metavar='FILE',
                    help='print outputs into a pretty format')
args = parser.parse_args()

dict = dictionary.Dictionary()
scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())

def dictolist(d):
    a = sorted(d.items(), key=lambda i: i[0])
    return [i[1] for i in a]

def load_sys(paths):
    src, tgt, hypos, log_probs = {}, {}, {}, {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.startswith(('S-', 'T-', 'H-')):
                    i = int(line[line.find('-')+1:line.find('\t')])
                    if line.startswith('S-'):
                        src[i] = line.split('\t')[1]
                    if line.startswith('T-'):
                        tgt[i] = line.split('\t')[1]
                    if line.startswith('H-'):
                        if i not in hypos:
                            hypos[i] = []
                            log_probs[i] = []
                        hypos[i].append(line.split('\t')[2])
                        log_probs[i].append(float(line.split('\t')[1]))
    return dictolist(src), dictolist(tgt), dictolist(hypos), dictolist(log_probs)

def load_ref(path):
    with open(path) as f:
        lines = f.readlines()
    src, tgt, refs = [], [], []
    i = 0
    while i < len(lines):
        if lines[i].startswith('S-'):
            src.append(lines[i].split('\t')[1])
            i += 1
        elif lines[i].startswith('T-'):
            tgt.append(lines[i].split('\t')[1])
            i += 1
        else:
            a = []
            while i < len(lines) and lines[i].startswith('R'):
                a.append(lines[i].split('\t')[1])
                i += 1
            refs.append(a)
    return src, tgt, refs

def merge(src, tgt, hypos, log_probs, path):
    with open(path, 'w') as f:
        for s, t, hs, lps in zip(src, tgt, hypos, log_probs):
            f.write(s)
            f.write(t)
            f.write('\n')
            for h, lp in zip(hs, lps):
                f.write('%f\t' % lp + h)
            f.write('------------------------------------------------------\n')

def corpus_bleu(ref, hypo):
    scorer.reset()
    for r, h in zip(ref, hypo):
        r_tok = tokenizer.Tokenizer.tokenize(r, dict)
        h_tok = tokenizer.Tokenizer.tokenize(h, dict)
        scorer.add(r_tok, h_tok)
    return scorer.score()

def sentence_bleu(ref, hypo):
    scorer.reset(one_init=True)
    r_tok = tokenizer.Tokenizer.tokenize(ref, dict)
    h_tok = tokenizer.Tokenizer.tokenize(hypo, dict)
    scorer.add(r_tok, h_tok)
    return scorer.score()

def pairwise(sents):
    _ref, _hypo = [], []
    for s in sents:
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    _ref.append(s[i])
                    _hypo.append(s[j])
    return corpus_bleu(_ref, _hypo)

def multi_ref(refs, hypos):
    _ref, _hypo = [], []
    ref_cnt = 0
    for rs, hs in zip(refs, hypos):
        a = set()
        for h in hs:
            s = [sentence_bleu(r, h) for r in rs]
            j = np.argmax(s)
            _ref.append(rs[j])
            _hypo.append(h)
            best = [k for k in range(len(rs)) if s[k] == s[j]]
            a.add(random.choice(best))
        ref_cnt += len(a)
    print('avg oracle BLEU: %.2f' % corpus_bleu(_ref, _hypo))
    print('#refs covered: %.2f' % (ref_cnt / len(refs)))

def intra_ref(refs):
    print('ref pairwise BLEU: %.2f' % pairwise(refs))
    _ref, _hypo = [], []
    for rs in refs:
        for i, h in enumerate(rs):
            rest = rs[:i] + rs[i+1:]
            s = [sentence_bleu(r, h) for r in rest]
            j = np.argmax(s)
            _ref.append(rest[j])
            _hypo.append(h)
    print('ref avg oracle BLEU (leave-one-out): %.2f' % corpus_bleu(_ref, _hypo))

if __name__ == '__main__':
    if args.sys:
        src, tgt, hypos, log_probs = load_sys(args.sys)
        print('pairwise BLEU: %.2f' % pairwise(hypos))
        if args.output:
            merge(src, tgt, hypos, log_probs, args.output)
    if args.ref:
        _, _, refs = load_ref(args.ref)
        if args.sys:
            multi_ref(refs, hypos)
        else:
            intra_ref(refs)

