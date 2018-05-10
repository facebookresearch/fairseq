#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import sys
import torch
from collections import namedtuple
from torch.autograd import Variable

from fairseq import options, tokenizer, utils
from fairseq.data.data_utils import collate_tokens
from fairseq.data.consts import LEFT_PAD_SOURCE
from fairseq.sequence_generator import SequenceGenerator

Batch = namedtuple('Batch', 'srcs tokens lengths')
Translation = namedtuple('Translation', 'src_str hypos alignments')


def buffered_read(buffer_size):
    buffer = []
    for src_str in sys.stdin:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, batch_size, src_dict):
    tokens = [tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long() for src_str in lines]
    lengths = [t.numel() for t in tokens]

    indices = np.argsort(lengths)
    num_batches = np.ceil(len(indices) / batch_size)
    batches = np.array_split(indices, num_batches)
    for batch_idxs in batches:
        batch_toks = [tokens[i] for i in batch_idxs]
        batch_toks = collate_tokens(batch_toks, src_dict.pad(), src_dict.eos(), LEFT_PAD_SOURCE,
                                    move_eos_to_beginning=False)
        yield Batch(
            srcs=[lines[i] for i in batch_idxs],
            tokens=batch_toks,
            lengths=tokens[0].new([lengths[i] for i in batch_idxs]),
        ), batch_idxs


def main(args):
    print(args)
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    if args.buffer_size < 1:
        args.buffer_size = 1

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(',')
    models, model_args = utils.load_ensemble_for_inference(model_paths, data_dir=args.data)
    src_dict, dst_dict = models[0].src_dict, models[0].dst_dict

    print('| [{}] dictionary: {} types'.format(model_args.source_lang, len(src_dict)))
    print('| [{}] dictionary: {} types'.format(model_args.target_lang, len(dst_dict)))

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        )

    # Initialize generator
    translator = SequenceGenerator(
        models, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
        unk_penalty=args.unkpen, sampling=args.sampling, sampling_topk=args.sampling_topk,
        minlen=args.min_len)

    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    def make_result(src_str, hypos):
        result = Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu(),
                align_dict=align_dict,
                dst_dict=dst_dict,
                remove_bpe=args.remove_bpe,
            )
            result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.alignments.append('A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment))))
        return result

    def process_batch(batch):
        tokens = batch.tokens
        lengths = batch.lengths

        if use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        translations = translator.generate(
            Variable(tokens),
            Variable(lengths),
            maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
        )

        return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    for inputs in buffered_read(args.buffer_size):
        indices = []
        results = []
        for batch, batch_indices in make_batches(inputs, max(1, args.max_sentences or 1), src_dict):
            indices.extend(batch_indices)
            results += process_batch(batch)

        for i in np.argsort(indices):
            result = results[i]
            print(result.src_str)
            for hypo, align in zip(result.hypos, result.alignments):
                print(hypo)
                print(align)


if __name__ == '__main__':
    parser = options.get_generation_parser(interactive=True)
    args = parser.parse_args()
    main(args)
