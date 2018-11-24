#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Evaluate the perplexity of a trained language model.
"""

import json
import math
import mosestokenizer as mt
import regex as re
from os import path
import subprocess
import tempfile
import torch

from fairseq import options, progress_bar, tasks, utils


def eval_dataset(task, model, dataset, data_file, use_cuda=True):
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 4096,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(model.max_positions()),
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        ignore_invalid_inputs=False,
    ).next_epoch_itr(shuffle=False)

    res = {}
    na_probs = {}

    was_training = model.training
    model.eval()

    detokenize = mt.MosesDetokenizer('en')

    with torch.no_grad(), progress_bar.build_progress_bar(args, itr) as t:
        for batch in t:
            if use_cuda:
                batch = utils.move_to_cuda(batch)
            start_res, end_res = model(**batch['net_input'])

            for start_preds, end_preds, id, text, mask in zip(start_res.cpu(), end_res.cpu(), batch['squad_ids'],
                                                              batch['net_input']['text'],
                                                              batch['net_input']['paragraph_mask']):

                mnz = mask.nonzero()
                sep_idx = mnz[0]
                last_idx = mnz[-1]

                maxlen = 30
                lenpen = 0.6

                na_score = start_preds[sep_idx] + end_preds[sep_idx]
                best = float('-inf')
                start_ind = sep_idx
                end_ind = sep_idx

                # for curr_start in range(sep_idx + 1, last_idx):
                #     for curr_end in range(curr_start + 1, min(last_idx + 1, curr_start + 2 + maxlen)):
                #         score = start_preds[curr_start] + end_preds[curr_end] - (curr_end - curr_start - 1) ** lenpen
                #         if score > best:
                #             best = score
                #             start_ind = curr_start
                #             end_ind = curr_end

                curr_start = sep_idx + 1
                curr_end = curr_start
                while curr_end < last_idx:
                    se = curr_end + 1
                    ee = min(max(curr_start + maxlen, se), last_idx) + 1
                    curr_end = torch.argmax(end_preds[se:ee]) + se
                    ss = max(curr_end - maxlen, sep_idx + 1)
                    se = curr_end
                    curr_start = torch.argmax(start_preds[ss:se]) + ss

                    score = start_preds[curr_start] + end_preds[curr_end]
                    if score > best:
                        best = score
                        start_ind = curr_start
                        end_ind = curr_end
                        assert end_ind > start_ind

                score_diff = na_score - best
                if score_diff > -1.0977530479431152:
                    pred = ''
                else:
                    pred = task.dictionary.string(text[start_ind:end_ind]).split()
                    pred = detokenize(pred)
                    pred = re.sub(r'(\d+) ([-–]) (\d+)', r'\1\2\3', pred)
                    pred = re.sub(r"(.+) ' (.+)", r"\1'\2", pred)

                res[id] = pred
                na_probs[id] = score_diff.item()

    with tempfile.NamedTemporaryFile('w') as f, tempfile.NamedTemporaryFile('w') as na_f:
        json.dump(res, f)
        f.flush()
        json.dump(na_probs, na_f)
        na_f.flush()
        res = subprocess.check_output(
            ["python", "official_squad_eval.py", data_file, f.name, '--na-prob-file', na_f.name],
            cwd=path.dirname(path.realpath(__file__)))
        print(res.decode('utf-8'))

    if was_training:
        model.train()


def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'

    print(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    print('| loading model(s) from {}'.format(parsed_args.path))
    models, args = utils.load_ensemble_for_inference(parsed_args.path.split(':'), task)

    assert len(models) == 1

    model = models[0]
    if use_cuda:
        model.cuda()

    for arg in vars(parsed_args).keys():
        if arg not in {'concat_sentences_mode'}:
            setattr(args, arg, getattr(parsed_args, arg))
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    data_file = path.join(args.data, args.data_file)
    eval_dataset(task, model, task.dataset(args.gen_subset), data_file, use_cuda)


if __name__ == '__main__':
    parser = options.get_parser('Evaluate SQUAD', 'squad')
    options.add_common_eval_args(parser)
    options.add_dataset_args(parser, gen=True)
    parser.add_argument('--data-file', type=str, help='the json data file to score (assumed to be in data dir)')
    args = options.parse_args_and_arch(parser)
    main(args)
