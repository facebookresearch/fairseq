# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
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

    parser = argparse.ArgumentParser("Evaluate PPX metric of a transcript.")
    parser.add_argument('--asr-transcript', type=str,
                        help='Path to the transcript file.')
    parser.add_argument('--cut-id', action='store_true',
                        help='Whether cut the first token (typically a seq id)')
    parser.add_argument('--cut-tail', action='store_true',
                        help='Whether cut the last token (typically a speaker id)')

    parser.add_argument('--manifest', type=str, default=None)
    parser.add_argument('--prompts-description', type=str, default=None)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    lm = torch.hub.load(
        'pytorch/fairseq', 'transformer_lm.wmt19.en', tokenizer='moses', bpe='fastbpe')

    lm.eval().cuda()  # disable dropout

    if args.manifest is None and args.prompts_description is None:
        target_ids = None
    else:
        target_ids = get_target_sequences(
            args.manifest, args.prompts_description)

    with open(args.asr_transcript, 'r') as fin:
        lines = fin.readlines()

    if target_ids is not None:
        filtered = []
        for line in lines:
            line_id = line.split()[-1]
            line_id = int(line_id.split('-')[1][:-1])
            if line_id in target_ids:
                filtered.append(line)
        lines = filtered
    else:
        pass

    if args.cut_id:
        lines = [' '.join(x.split()[1:]) for x in lines]
    if args.cut_tail:
        lines = [' '.join(x.split()[:-1]) for x in lines]
    lines = [x.strip().lower() for x in lines]

    def get_logprob(sent): return \
        lm.score(sent)['positional_scores'].mean().neg().item()

    logprobs = [get_logprob(l) for l in lines]

    filtered = [x for x in logprobs if not np.isnan(x)]
    if len(filtered) != len(logprobs):
        warnings.warn("NaNs detected!")
        logprobs = filtered

    perplexities = [np.exp(l) for l in logprobs]

    for name, stats in [('logprob', logprobs), ('perplexity', perplexities)]:
        mean = np.mean(stats)
        sem = np.std(stats) / np.sqrt(len(stats))

        median = np.median(stats)
        interval = list(np.percentile(stats, [10, 90]))

        mean, sem, median, percentile10, percentile90 = [
            round(x, 2) for x in [mean, sem, median] + interval]

        print(name)
        print(f"\tMean {mean} +- {sem}")
        print(
            f"\tMedian {median}, 90% confidence interval {percentile10}...{percentile90}")


if __name__ == '__main__':
    main()
