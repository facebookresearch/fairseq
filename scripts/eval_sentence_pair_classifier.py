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


def eval_dataset(task, model, dataset, out_file, labels, use_cuda=True):
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

    predictions = {}

    with torch.no_grad(), progress_bar.build_progress_bar(args, itr) as t:
        for batch in t:
            if use_cuda:
                batch = utils.move_to_cuda(batch)

            logits = model(**batch['net_input'])

            if logits.shape[-1] == 1:
                pred_class = logits.cpu()
            else:
                pred_class = logits.argmax(dim=-1).cpu()

            for i in range(len(pred_class)):
                predictions[batch['id'][i].item()] = pred_class[i].item()

    with open(out_file, 'w') as out_f:
        print('index\tprediction', file=out_f)
        for i in range(len(predictions)):
            print(f'{i}\t{labels[predictions[i]]}', file=out_f)


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
        if arg not in ('concat_sentences_mode'):
            setattr(args, arg, getattr(parsed_args, arg))

    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    eval_dataset(task, model, task.dataset(args.gen_subset), args.out_file, args.labels, use_cuda)


if __name__ == '__main__':
    parser = options.get_parser('Evaluate Sentence Pair Classifier', 'sentence_pair_classification')
    options.add_common_eval_args(parser)
    options.add_dataset_args(parser, gen=True)
    parser.add_argument('--out-file', type=str, help='output filename')

    parser.add_argument(
        '--labels',
        default=['neutral', 'entailment', 'contradiction'],
        nargs='+',
        help='list of labels',
    )

    args = options.parse_args_and_arch(parser)
    main(args)
