# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict
import numpy as np
from misc.bleu_utils import sentence_bleu
import json
import warnings


def get_args():
    import argparse

    parser = argparse.ArgumentParser("Tool to calculate Continuation-BLEU2")
    parser.add_argument('--asr-transcript', type=str,
                        help='Path to the transcript file.')
    parser.add_argument('--prompts-description', type=str,
                        help='Path to the ground-truth continuation')
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--take-shortest', type=int, default=1000)

    args = parser.parse_args()

    return args


def main():
    # NLTK produces warnings
    warnings.filterwarnings("ignore")

    args = get_args()

    with open(args.prompts_description, 'r') as fin:
        original_continuations = json.loads(fin.read())

    sequence2length = [(k, v[0]) for k, v in original_continuations.items()]
    assert all(float(v) >= 6.0 for (_, v) in sequence2length)  # 6 seconds

    sequence2length.sort(key=lambda x: x[1])
    to_take = set(v[0] for v in sequence2length[:args.take_shortest])

    with open(args.manifest, 'r') as fin:
        fin.readline()

        linenum2file = dict([
            (i, l.split("__")[0]) for (i, l) in enumerate(fin)
        ])

    max_files = max(linenum2file.keys())
    continuations = defaultdict(list)

    mean_length_after = 0
    n_examples = 0

    with open(args.asr_transcript, 'r') as fin:
        for line in fin:
            n_examples += 1
            line = line.split()
            sequence_id = int(line[-1].split('-')[1][:-1])

            assert sequence_id <= max_files

            sequence_name = linenum2file[sequence_id]

            continuations[sequence_name].append(line[:-1])
            mean_length_after += len(line)

    mean_length_after /= n_examples
    print(f'Mean length of continuations, in words: {mean_length_after}')
    metric_values = []

    mean_ground_truth_words = 0
    n_examples = 0
    n_candidates = 0

    for k, candidates in continuations.items():
        if k not in to_take:
            continue

        n_examples += 1

        ground_truth = original_continuations[k][1].split()
        n_candidates += len(candidates)
        bleu = sentence_bleu(candidates, ground_truth, weights=(
            0.5, 0.5), no_length_penalty=True, averaging_mode="geometric")
        mean_ground_truth_words += len(ground_truth)

        metric_values.append(bleu)

    n = len(metric_values)
    print(
        f'Median BLEU over {n} examples: {np.median(metric_values)} +- {np.std(metric_values) / np.sqrt(n)}')


if __name__ == '__main__':
    main()
