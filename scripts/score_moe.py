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
from itertools import chain
import sys
import numpy as np
import random

from sacrebleu import compute_bleu, corpus_bleu as _corpus_bleu


def main():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--sys', nargs='*', default='', metavar='FILE',
                        help='path to system output')
    parser.add_argument('--ref', default='', metavar='FILE',
                        help='path to references')
    parser.add_argument('--output', default='', metavar='FILE',
                        help='print outputs into a pretty format')
    args = parser.parse_args()

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


def dictolist(d):
    a = sorted(d.items(), key=lambda i: i[0])
    return [i[1] for i in a]


def load_sys(paths):
    src, tgt, hypos, log_probs = {}, {}, {}, {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.rstrip()
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
            src.append(lines[i].split('\t')[1].rstrip())
            i += 1
        elif lines[i].startswith('T-'):
            tgt.append(lines[i].split('\t')[1].rstrip())
            i += 1
        else:
            a = []
            while i < len(lines) and lines[i].startswith('R'):
                a.append(lines[i].split('\t')[1].rstrip())
                i += 1
            refs.append(a)
    return src, tgt, refs


def merge(src, tgt, hypos, log_probs, path):
    with open(path, 'w') as f:
        for s, t, hs, lps in zip(src, tgt, hypos, log_probs):
            f.write(s + '\n')
            f.write(t + '\n')
            f.write('\n')
            for h, lp in zip(hs, lps):
                f.write('\t%f\t%s\n' % (lp, h.strip()))
            f.write('------------------------------------------------------\n')


def corpus_bleu(sys_stream, ref_streams):
    bleu = _corpus_bleu(sys_stream, ref_streams, tokenize='none')
    return bleu.score


def sentence_bleu(hypothesis, reference):
    bleu = _corpus_bleu(hypothesis, reference)
    for i in range(1, 4):
        bleu.counts[i] += 1
        bleu.totals[i] += 1
    bleu = compute_bleu(
        bleu.counts, bleu.totals,
        bleu.sys_len, bleu.ref_len,
        smooth='exp', smooth_floor=0.0,
    )
    return bleu.score


def pairwise(sents):
    _ref, _hypo = [], []
    for s in sents:
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    _ref.append(s[i])
                    _hypo.append(s[j])
    return corpus_bleu(_hypo, [_ref])


def multi_ref(refs, hypos):
    _ref, _hypo = [], []
    ref_cnt = 0
    assert len(refs) == len(hypos)

    # count number of refs covered
    for rs, hs in zip(refs, hypos):
        a = set()
        for h in hs:
            s = [sentence_bleu(h, r) for r in rs]
            j = np.argmax(s)
            _ref.append(rs[j])
            _hypo.append(h)
            best = [k for k in range(len(rs)) if s[k] == s[j]]
            a.add(random.choice(best))
        ref_cnt += len(a)
    print('#refs covered: %.2f' % (ref_cnt / len(refs)))

    # transpose refs and hypos
    refs = list(zip(*refs))
    hypos = list(zip(*hypos))

    # compute average corpus BLEU
    k = len(hypos)
    m = len(refs)
    concat_hypos = []
    concat_refs = [[] for j in range(m - 1)]
    for i in range(m):
        concat_hypos.append([h for hs in hypos for h in hs])
        rest = refs[:i] + refs[i+1:]
        for j in range(m - 1):
            concat_refs[j].extend(rest[j] * k)
    concat_hypos = list(chain.from_iterable(concat_hypos))
    bleu = corpus_bleu(concat_hypos, concat_refs)
    print('multi-reference BLEU (leave-one-out): %.2f' % bleu)


def intra_ref(refs):
    print('ref pairwise BLEU: %.2f' % pairwise(refs))
    refs = list(zip(*refs))
    m = len(refs)
    concat_h = []
    concat_rest = [[] for j in range(m - 1)]
    for i, h in enumerate(refs):
        rest = refs[:i] + refs[i+1:]
        concat_h.append(h)
        for j in range(m - 1):
            concat_rest[j].extend(rest[j])
    concat_h = list(chain.from_iterable(concat_h))
    bleu = corpus_bleu(concat_h, concat_rest)
    print('multi-reference BLEU (leave-one-out): %.2f' % bleu)


if __name__ == '__main__':
    main()
