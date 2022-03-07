# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nltk
from misc.bleu_utils import sentence_bleu
import warnings


def get_target_sequences(manifest, ground_truth, to_take=1000):
    import json
    import pathlib

    with open(ground_truth, 'r') as fin:
        original_continuations = json.loads(fin.read())

    sequence2length = [(k, v[0]) for k, v in original_continuations.items()]
    assert all(float(v) >= 6.0 for (_, v) in sequence2length)  # 6 seconds

    sequence2length.sort(key=lambda x: x[1])
    to_take_sequences = set(v[0] for v in sequence2length[:to_take])
    to_take_ids = []

    with open(manifest, 'r') as f:
        f.readline()

        for i, line in enumerate(f.readlines()):
            seq_id = line.split()[0]
            seq_id = pathlib.Path(seq_id).name.split('__')[0]

            if seq_id in to_take_sequences:
                to_take_ids.append(i)

    print(f'Took {len(to_take_ids)} ids')
    return set(to_take_ids)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--asr-transcript', type=str,
                        help='Path to the transcript file.')

    parser.add_argument('--manifest', required=True)
    parser.add_argument('--prompts-description', required=True)

    parser.add_argument('--cut-id', action='store_true',
                        help='Whether cut the first token (typically a seq id)')
    parser.add_argument('--cut-tail', action='store_true',
                        help='Whether cut the last token (typically a speaker id)')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    return args


def get_self_bleu(utterances, averaging_mode, weights):
    self_bleu = []

    for i in range(len(utterances)):
        hypo = utterances[i]
        rest = utterances[:i] + utterances[i+1:]

        self_bleu.append(sentence_bleu(rest, hypo, weights,
                         no_length_penalty=True, averaging_mode=averaging_mode))

    return self_bleu


def get_self_bleu2_arithmetic(utterances):
    weights = (0.5, 0.5)  # equal weight for unigrams and bigrams
    return get_self_bleu(utterances, averaging_mode='arithmetic', weights=weights)


def get_self_bleu2_geometric(utterances):
    weights = (0.5, 0.5)
    return get_self_bleu(utterances, averaging_mode='geometric', weights=weights)


def get_auto_bleu2_arithmetic(utterances):
    weights = (0.5, 0.5)
    return [auto_bleu(u, mean_mode='arithmetic', weights=weights) for u in utterances]


def get_auto_bleu2_geometric(utterances):
    weights = (0.5, 0.5)
    return [auto_bleu(u, mean_mode='geometric', weights=weights) for u in utterances]


def get_auto_bleu3_geometric(utterances):
    weights = (1./3, 1./3, 1./3)
    return [auto_bleu(u, mean_mode='geometric', weights=weights) for u in utterances]


def get_auto_bleu3_arithmetic(utterances):
    weights = (1./3, 1./3, 1./3)
    return [auto_bleu(u, mean_mode='arithmetic', weights=weights) for u in utterances]


def get_self_bleu3_arithmetic(utterances):
    weights = (1./3, 1./3, 1./3)
    return get_self_bleu(utterances, averaging_mode='arithmetic', weights=weights)


def get_self_bleu3_geometric(utterances):
    weights = (1./3, 1./3, 1./3)
    return get_self_bleu(utterances, averaging_mode='geometric', weights=weights)


def auto_bleu(sentence, weights, mean_mode='arithmetic'):
    if len(sentence) <= 1:
        return 0

    N = len(weights)

    bleu_n = np.zeros([N])
    for n in range(N):
        targ_ngrams = list(nltk.ngrams(sentence, n+1))
        for p in range(len(targ_ngrams)):
            left = sentence[:p]
            right = sentence[(p+n+1):]
            rest_ngrams = list(nltk.ngrams(left, n+1)) + \
                list(nltk.ngrams(right, n+1))
            # compute the nb of matching ngrams
            bleu_n[n] += targ_ngrams[p] in rest_ngrams
        bleu_n[n] /= len(targ_ngrams)  # average them to get a proportion

    weights = np.array(weights)
    if mean_mode == 'arithmetic':
        return (bleu_n * weights).sum()
    elif mean_mode == 'geometric':
        return (bleu_n ** weights).prod()
    else:
        raise ValueError(f'Unknown agggregation mode {mean_mode}')


def main():
    from multiprocessing import Pool

    args = get_args()
    target_ids = get_target_sequences(args.manifest, args.prompts_description)

    with open(args.asr_transcript, 'r') as fin:
        lines = fin.readlines()

    terms = [x.strip().split() for x in lines]
    filtered = []
    for term in terms:
        line_id = int(term[-1].split('-')[1][:-1])
        if line_id in target_ids:
            filtered.append(term)
    terms = filtered

    if args.cut_id:
        terms = [x[1:] for x in terms]
    if args.cut_tail:
        terms = [x[:-1] for x in terms]

    if args.debug:
        terms = terms[:10]

    tasks = [
        ('Self-BLEU2-arithmetic', get_self_bleu2_arithmetic),
        ('Self-BLEU2-geometric', get_self_bleu2_geometric),
        ('Auto-BLEU2-arithmetic', get_auto_bleu2_arithmetic),
        ('Auto-BLEU2-geometric', get_auto_bleu2_geometric),

        ('Self-BLEU3-arithmetic', get_self_bleu3_arithmetic),
        ('Self-BLEU3-geometric', get_self_bleu3_geometric),
        ('Auto-BLEU3-arithmetic', get_auto_bleu3_arithmetic),
        ('Auto-BLEU3-geometric', get_auto_bleu3_geometric),
    ]

    n_processes = min(16, len(tasks))
    with Pool(n_processes) as pool:
        metrics = pool.map(run_f, [(t[1], terms) for t in tasks])

    for (metric_name, _), metric in zip(tasks, metrics):
        metric, sem = np.mean(metric), np.std(metric) / np.sqrt(len(metric))

        metric, sem = [
            round(100 * x, 2) for x in [metric, sem]
        ]

        print(f'{metric_name} {metric} +- {sem}')


def run_f(task_params):
    f, terms = task_params
    return f(terms)


if __name__ == '__main__':
    # NLTK produces warnings
    warnings.filterwarnings("ignore")

    main()
