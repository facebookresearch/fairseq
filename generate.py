#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch

from fairseq import bleu, data, options, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.progress_bar import progress_bar
from fairseq.sequence_generator import SequenceGenerator


def main():
    parser = options.get_parser('Generation')
    parser.add_argument('--path', metavar='FILE', required=True, action='append',
                        help='path(s) to model file(s)')
    dataset_args = options.add_dataset_args(parser)
    dataset_args.add_argument('--batch-size', default=32, type=int, metavar='N',
                              help='batch size')
    dataset_args.add_argument('--gen-subset', default='test', metavar='SPLIT',
                              help='data subset to generate (train, valid, test)')
    options.add_generation_args(parser)

    args = parser.parse_args()
    print(args)

    if args.no_progress_bar:
        progress_bar.enabled = False
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset
    dataset = data.load_dataset(args.data, [args.gen_subset], args.source_lang, args.target_lang)
    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args
        args.source_lang, args.target_lang = dataset.src, dataset.dst

    # Load ensemble
    print('| loading model(s) from {}'.format(', '.join(args.path)))
    models, _ = utils.load_ensemble_for_inference(args.path, dataset.src_dict, dataset.dst_dict)

    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(dataset.splits[args.gen_subset])))

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam)

    # Initialize generator
    translator = SequenceGenerator(
        models, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
        unk_penalty=args.unkpen)
    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Generate and compute BLEU score
    scorer = bleu.Scorer(dataset.dst_dict.pad(), dataset.dst_dict.eos(), dataset.dst_dict.unk())
    max_positions = min(model.max_encoder_positions() for model in models)
    itr = dataset.eval_dataloader(
        args.gen_subset, batch_size=args.batch_size, max_positions=max_positions,
        skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test)
    num_sentences = 0
    with progress_bar(itr, smoothing=0, leave=False) as t:
        wps_meter = TimeMeter()
        gen_timer = StopwatchMeter()
        translations = translator.generate_batched_itr(
            t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
            cuda_device=0 if use_cuda else None, timer=gen_timer)
        for sample_id, src_tokens, target_tokens, hypos in translations:
            # Process input and ground truth
            target_tokens = target_tokens.int().cpu()
            src_str = dataset.src_dict.string(src_tokens, args.remove_bpe)
            target_str = dataset.dst_dict.string(target_tokens, args.remove_bpe, escape_unk=True)
            if not args.quiet:
                print('S-{}\t{}'.format(sample_id, src_str))
                print('T-{}\t{}'.format(sample_id, target_str))

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu(),
                    align_dict=align_dict,
                    dst_dict=dataset.dst_dict,
                    remove_bpe=args.remove_bpe)

                if not args.quiet:
                    print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                    print('A-{}\t{}'.format(sample_id, ' '.join(map(str, alignment))))

                # Score only the top hypothesis
                if i == 0:
                    if args.remove_bpe is not None:
                        # Convert the string without BPE back to tokens for evaluation
                        target_tokens = tokenizer.Tokenizer.tokenize(target_str,
                                                                     dataset.dst_dict,
                                                                     add_if_not_exist=True)
                    scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(src_tokens.size(0))
            t.set_postfix(wps='{:5d}'.format(round(wps_meter.avg)), refresh=False)
            num_sentences += 1

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))
    print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))


if __name__ == '__main__':
    main()
