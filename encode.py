#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Encode raw text with a trained model. Batches data on-the-fly.
"""

import numpy as np

import torch as th

from interactive import buffered_read, make_batches
from fairseq import checkpoint_utils, options, tasks, utils


def encode(model, src_tokens, src_lengths):
    """Encode a batch of sentences"""
    model.eval()
    # Run encoder
    encoder_out = model.encoder(src_tokens, src_lengths=src_lengths)
    encodings = encoder_out["encoder_out"]
    # Average along the length dimension (be wary of different lengths)
    bsz, L = src_tokens.size()
    src_lengths = src_lengths.float()
    # Create a mask with 0s where padding tokens are located
    positions = th.arange(L).view(-1, 1).repeat(1, bsz).float()
    mask = positions.to(src_lengths.device).lt(src_lengths.view(1, -1))
    # Multiply by mask and sum over length dimension
    mean_encodings = th.einsum("lb,lbd->bd", [mask.float(), encodings])
    # Normalize by respective lengths
    mean_encodings /= src_lengths.view(-1, 1)
    # Detach and return
    return mean_encodings.detach()


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = th.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    [model], _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Hack to support GPT-2 BPE
    if args.remove_bpe == 'gpt2':
        from fairseq.gpt2_bpe.gpt2_encoding import get_encoder
        decoder = get_encoder(
            'fairseq/gpt2_bpe/encoder.json',
            'fairseq/gpt2_bpe/vocab.bpe',
        )
        def encode_fn(x): return ' '.join(map(str, decoder.encode(x)))
    else:
        decoder = None
        def encode_fn(x): return x
    # Max position for batching
    max_positions = utils.resolve_max_positions(
        task.max_positions(), model.max_positions()
    )
    # Prompt
    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_idx = 0
    # This tracks all encodings in the order that they are given as input
    all_encodings = []
    # Read chunks of the input stream one at a time
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        # Make batches on the fly
        for batch in make_batches(inputs, args, task, max_positions):
            # Retrieve inputs
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
            # Encode
            encodings = encode(model, src_tokens, src_lengths)
            # Save encodings in the correct order
            # (the batches are out of order to optimize padding)
            for i, (idx, h) in enumerate(zip(batch.ids.tolist(), encodings)):
                results.append((start_idx + idx, h))
        # Save the encodings in order
        for _, h in sorted(results, key=lambda x: x[0]):
            all_encodings.append(h.cpu().numpy())
        # update running id counter
        start_idx += len(inputs)
    # Save all encodings to npy
    np.save(args.output_file, np.stack(all_encodings))


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
