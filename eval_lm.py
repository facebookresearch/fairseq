#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import options, utils, progress_bar
from fairseq.data import data_utils, data_loaders
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer


def main(args):
    assert args.path is not None, '--path required for evaluation!'
    print(args)

    if args.max_target_positions is None:
        args.max_target_positions = 1024

    use_cuda = torch.cuda.is_available() and not args.cpu
    dataset = data_loaders.load_dataset(args, [args.gen_subset], False)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(','), dataset.src_dict, dataset.dst_dict)

    print('| Dictionary: {} types'.format(len(dataset.src_dict)))
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(dataset.splits[args.gen_subset])))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        model.src_dict = dataset.src_dict
        model.dst_dict = dataset.dst_dict

    itr = dataset.eval_dataloader(
        args.gen_subset,
        max_sentences=args.max_sentences or 4,
        max_positions=args.max_target_positions or 1024,
        descending=True,
    )
    itr = data_utils.ShardedIterator(itr, args.num_shards, args.shard_id)

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(models)
    if use_cuda:
        scorer.cuda()

    score_sum = 0.
    count = 0
    with progress_bar.build_progress_bar(args, itr) as t:
        results = scorer.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
        wps_meter = TimeMeter()
        for _, src_tokens, __, hypos in results:
            for hypo in hypos:
                pos_scores = hypo['positional_scores']
                inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                if inf_scores.any():
                    print('| Skipping tokens with inf scores:',
                          dataset.src_dict.string(hypo['tokens'][inf_scores.nonzero()]))
                    pos_scores = pos_scores[(~inf_scores).nonzero()]
                score_sum += pos_scores.sum()
                count += pos_scores.numel()
            wps_meter.update(src_tokens.size(0))
            t.log({'wps': round(wps_meter.avg)})

    avg_nll_loss = -score_sum / count
    print('| Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))
    print('| Loss: {:.4f}, Perplexity: {:.2f}'.format(avg_nll_loss, np.exp(avg_nll_loss)))


if __name__ == '__main__':
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
