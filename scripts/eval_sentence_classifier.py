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

from os import path

import torch

from fairseq import options, progress_bar, tasks, utils
from fairseq.meters import ClassificationMeter


def eval_dataset(task, model, dataset, out_file, thresholds, compute_metrics, use_cuda=True):
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 4096,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(model.max_positions()),
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        ignore_invalid_inputs=False,
    ).next_epoch_itr(shuffle=False)

    model.eval()

    if thresholds is None or len(thresholds) == 0:
        thresholds = [0.]

    predictions = {}
    meters = {t: ClassificationMeter() for t in thresholds}

    with torch.no_grad(), progress_bar.build_progress_bar(args, itr) as t:
        for batch in t:
            if use_cuda:
                batch = utils.move_to_cuda(batch)

            logits = model(**batch['net_input'])
            assert logits.shape[-1] == 2

            normed = logits.softmax(dim=-1)
            diff = normed[:, :, 0] - normed[:, :, 1]
            pred_class = {t: (diff < t) for t in thresholds}

            for i in range(len(batch['id'])):
                pred = pred_class[thresholds[0]][i].item()
                predictions[batch['id'][i].item()] = pred
                if compute_metrics:
                    preds = {t: pred_class[t][i].item() for t in thresholds} if len(thresholds) > 0 else {
                        0.: pred_class[i].item()}
                    actual = batch['target'][i].item()
                    assert actual in {0, 1}, 'bad actual! ' + str(actual)
                    for t, p in preds.items():
                        assert p in {0, 1}, 'bad p! ' + str(p)
                        tp = int(actual == 1 and p == 1)
                        fp = int(actual == 0 and p == 1)
                        tn = int(actual == 0 and p == 0)
                        fn = int(actual == 1 and p == 0)
                        assert tp + fp + tn + fn == 1
                        meters[t].update(tp, tn, fp, fn)

    if compute_metrics:
        for t, m in meters.items():
            print('{}: {}'.format(t, m.vals()))

    if out_file is not None:
        with open(out_file, 'w') as out_f:
            print('index\tprediction', file=out_f)
            for i in range(len(predictions)):
                print(f'{i}\t{predictions[i]}', file=out_f)


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
        setattr(args, arg, getattr(parsed_args, arg))
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    eval_dataset(task, model, task.dataset(args.gen_subset), args.out_file, args.thresholds,
                 args.compute_metrics, use_cuda)


if __name__ == '__main__':
    parser = options.get_parser('Evaluate Single Sentence Classifier', 'sentence_classification')
    options.add_common_eval_args(parser)
    options.add_dataset_args(parser, gen=True)
    parser.add_argument('--out-file', type=str, help='output filename')
    parser.add_argument('--thresholds', nargs='+', type=float, help='thresholds to try or use')
    parser.add_argument('--compute-metrics', action='store_true',
                        help='if set, uses the labels to compute metrics for each threshold')
    args = options.parse_args_and_arch(parser)
    main(args)
