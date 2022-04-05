#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Sample from a trained LM; hacked fairseq-interactive
"""
from collections import namedtuple
import os
import ast
import numpy as np

from fairseq import checkpoint_utils, options, tasks, utils

import tqdm

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def make_batches(lines, args, task, max_positions):
    tokens = [
        task.source_dictionary.encode_line(
            src_str, add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.dataset.max_tokens,
        max_sentences=args.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.dataset.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def main(args):
    arg_prompts = args.prompts
    arg_output = args.output
    arg_debug = args.debug
    arg_sample_size = args.samples_per_prompt

    try:
        from fairseq.dataclass.utils import convert_namespace_to_omegaconf
        args = convert_namespace_to_omegaconf(args)
    except:
        pass

    # if args.max_tokens is None and args.max_sentences is None:
    if args.common.seed is not None:
        np.random.seed(args.common.seed)
        utils.set_torch_seed(args.common.seed)

    if args.generation.sampling:
        args.generation.nbest = args.generation.beam = arg_sample_size

    task = tasks.setup_task(args.task)

    overrides = ast.literal_eval(args.common_eval.model_overrides)

    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.common_eval.path.split(os.pathsep),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    output_file = open(arg_output, 'w')

    with open(arg_prompts, 'r') as fin:
        lines = fin.readlines()

    split = [x.split('|', 1) for x in lines]
    seq_id = [x[0] for x in split]
    prompts = [x[1] for x in split]

    if args.generation.prefix_size >= 0:
        prompts = [' '.join(l.split()[:args.generation.prefix_size])
                   for l in prompts]

    if arg_debug:
        prompts = prompts[:10]

    generator = task.build_generator(models, args.generation)

    start_id = 0
    pbar = tqdm.tqdm(total=len(prompts))
    for batch in make_batches(prompts, args, task, max_positions):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        src_tokens = src_tokens.cuda()
        src_lengths = src_lengths.cuda()

        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
        }

        results = []
        translations = task.inference_step(generator, models, sample)
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            results.append((i + start_id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(
                    src_tokens, args.common_eval.post_process)

            # Process top predictions
            for hypo_id, hypo in enumerate(hypos):
                _hypo_tokens, hypo_str, _alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.common_eval.post_process,
                )

                detok_hypo_str = hypo_str
                utterance = detok_hypo_str
                print(f'{seq_id[id]}__{hypo_id}|{utterance}', file=output_file)
            pbar.update(1)
        start_id += len(results)

    # output_file.close()


def cli_main():
    parser = options.get_interactive_generation_parser()
    parser.add_argument('--prompts', type=str, default=None, required=True)
    parser.add_argument('--output', type=str, default=None, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--samples-per-prompt', type=int, default=1)

    args = options.parse_args_and_arch(parser)

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    main(args)


if __name__ == '__main__':
    cli_main()
