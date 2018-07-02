#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import json

from fairseq import data, options, progress_bar, tasks, utils


def write_json(file, states):
    assert len(states) > 0

    sent_len = len(states[0])
    assert all(len(s) == sent_len for s in states[1:])

    sent_states = []
    for i in range(sent_len):
        sent_states.append([s[i] for s in states])

    json.dump(sent_states, file)

def main(args):
    """
    The purpose of this file is to read sentences from a file, run them through a trained language model and dump
    the language model hidden states into another file. These hidden states may then be used for other tasks, such
    as ELMo embeddings.

    Input is the path for the model checkpoint and the data directory path that contains the dictionary that
    the model was trained with.

    The output is a json file that, for each line in the input file, contains a json array. The json array
    contains an array of hidden states which may be a flat array or contain tuples for forward/backward states
    (depending on the --flatten option).

    """
    assert args.path is not None, '--path required for evaluation!'

    args.raw_text = True

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    task = tasks.setup_task(args)
    task.load_dataset(args.input, full_path=True)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task)

    assert len(models) == 1, 'ensembles are currently not supported'
    model = models[0]

    model.eval()
    model.make_generation_fast_()
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    itr = data.EpochBatchIterator(
        dataset=task.dataset(args.input),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences or 4,
        max_positions=model.max_positions(),
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        ignore_invalid_inputs=True,
    ).next_epoch_itr(shuffle=False)

    with open(args.output, 'w') as out, torch.no_grad(), progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            s = utils.move_to_cuda(sample) if use_cuda else sample
            net_input = s['net_input']

            _, extra_out = model.forward(**net_input)
            assert 'inner_states' in extra_out

            bsz = net_input['src_tokens'].size(0)

            for i in range(bsz):
                sent_states = [x[:, i, :].tolist() for x in extra_out['inner_states']]
                write_json(out, sent_states)
                out.write('\n')


if __name__ == '__main__':
    parser = options.get_eval_lm_parser()

    parser.add_argument('--input', metavar='FILE', required=True,
                        help='Path to the input containing raw text data, one example per line')
    parser.add_argument('--output', metavar='FILE', required=True,
                        help='Path to the output into which hidden states will be written')

    args = options.parse_args_and_arch(parser)
    main(args)
