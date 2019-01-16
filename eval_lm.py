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

import numpy as np
import torch

from fairseq import options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.utils import import_user_module


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.word, self.count, self.log_prob, self.is_bpe,
                                               self.next_word_prob, self.count - self.missing_next_words)


def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'

    import_user_module(parsed_args)

    print(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    print('| loading model(s) from {}'.format(parsed_args.path))
    models, args = utils.load_ensemble_for_inference(
        parsed_args.path.split(':'), task, model_arg_overrides=eval(parsed_args.model_overrides),
    )

    for arg in vars(parsed_args).keys():
        if arg not in {'self_target', 'future_target', 'past_target', 'tokens_per_sample', 'output_size_dictionary'}:
            setattr(args, arg, getattr(parsed_args, arg))
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()

    assert len(models) > 0

    print('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(models, task.target_dictionary)
    if use_cuda:
        scorer.cuda()

    score_sum = 0.
    count = 0

    if args.remove_bpe is not None:
        bpe_cont = args.remove_bpe.rstrip()
        bpe_toks = set(i for i in range(len(task.dictionary)) if task.dictionary[i].endswith(bpe_cont))
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    with progress_bar.build_progress_bar(args, itr) as t:
        results = scorer.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
        wps_meter = TimeMeter()
        for _, src_tokens, __, hypos in results:
            for hypo in hypos:
                pos_scores = hypo['positional_scores']

                skipped_toks = 0
                if bpe_toks is not None:
                    for i in range(len(hypo['tokens']) - 1):
                        if hypo['tokens'][i].item() in bpe_toks:
                            skipped_toks += 1
                            pos_scores[i + 1] += pos_scores[i]
                            pos_scores[i] = 0

                inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                if inf_scores.any():
                    print('| Skipping tokens with inf scores:',
                          task.target_dictionary.string(hypo['tokens'][inf_scores.nonzero()]))
                    pos_scores = pos_scores[(~inf_scores).nonzero()]
                score_sum += pos_scores.sum().cpu()
                count += pos_scores.numel() - skipped_toks

                if args.output_word_probs or args.output_word_stats:
                    w = ''
                    word_prob = []
                    is_bpe = False
                    for i in range(len(hypo['tokens'])):
                        w_ind = hypo['tokens'][i].item()
                        w += task.dictionary[w_ind]
                        if bpe_toks is not None and w_ind in bpe_toks:
                            w = w[:-bpe_len]
                            is_bpe = True
                        else:
                            word_prob.append((w, pos_scores[i].item()))

                            next_prob = None
                            ind = i + 1
                            while ind < len(hypo['tokens']):
                                if pos_scores[ind].item() != 0:
                                    next_prob = pos_scores[ind]
                                    break
                                ind += 1

                            word_stats.setdefault(w, WordStat(w, is_bpe)).add(pos_scores[i].item(), next_prob)
                            is_bpe = False
                            w = ''
                    if args.output_word_probs:
                        print('\t'.join('{} [{:2f}]'.format(x[0], x[1]) for x in word_prob))

            wps_meter.update(src_tokens.size(0))
            t.log({'wps': round(wps_meter.avg)})

    avg_nll_loss = -score_sum / count
    print('| Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))
    print('| Loss: {:.4f}, Perplexity: {:.2f}'.format(avg_nll_loss, np.exp(avg_nll_loss)))

    if args.output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            print(ws)


if __name__ == '__main__':
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
