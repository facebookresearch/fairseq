#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import sys
import torch
from torch.autograd import Variable

from fairseq import bleu, data, options, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.progress_bar import progress_bar
from fairseq.sequence_generator import SequenceGenerator


def main():
    parser = options.get_parser('Generation')
    parser.add_argument('--path', metavar='FILE', required=True, action='append',
                        help='path(s) to model file(s)')
    dataset_args = options.add_dataset_args(parser)
    dataset_args.add_argument('-i', '--interactive', action='store_true',
                              help='generate translations in interactive mode')
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
    dataset = data.load_with_check(args.data, [args.gen_subset], args.source_lang, args.target_lang)
    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args
        args.source_lang, args.target_lang = dataset.src, dataset.dst

    # Load ensemble
    print('| loading model(s) from {}'.format(', '.join(args.path)))
    models = utils.load_ensemble_for_inference(args.path, dataset.src_dict, dataset.dst_dict)

    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    if not args.interactive:
        print('| {} {} {} examples'.format(args.data, args.gen_subset, len(dataset.splits[args.gen_subset])))

    # Max positions is the model property but it is needed in data reader to be able to
    # ignore too long sentences
    args.max_positions = min(args.max_positions, *(m.decoder.max_positions() for m in models))

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(not args.no_beamable_mm)

    # Initialize generator
    translator = SequenceGenerator(
        models, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen
    )
    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    align_dict = {}
    if args.unk_replace_dict != '':
        assert args.interactive, \
            'Unknown word replacement requires access to original source and is only supported in interactive mode'
        with open(args.unk_replace_dict, 'r') as f:
            for line in f:
                l = line.split()
                align_dict[l[0]] = l[1]

    def replace_unk(hypo_str, align_str, src, unk):
        hypo_tokens = hypo_str.split()
        src_tokens = tokenizer.tokenize_line(src)
        align_idx = [int(i) for i in align_str.split()]
        for i, ht in enumerate(hypo_tokens):
            if ht == unk:
                src_token = src_tokens[align_idx[i]]
                if src_token in align_dict:
                    hypo_tokens[i] = align_dict[src_token]
                else:
                    hypo_tokens[i] = src_token
        return ' '.join(hypo_tokens)

    def display_hypotheses(id, src, orig, ref, hypos):
        if args.quiet:
            return
        id_str = '' if id is None else '-{}'.format(id)
        src_str = dataset.src_dict.string(src, args.remove_bpe)
        print('S{}\t{}'.format(id_str, src_str))
        if orig is not None:
            print('O{}\t{}'.format(id_str, orig.strip()))
        if ref is not None:
            print('T{}\t{}'.format(id_str, dataset.dst_dict.string(ref, args.remove_bpe, escape_unk=True)))
        for hypo in hypos:
            hypo_str = dataset.dst_dict.string(hypo['tokens'], args.remove_bpe)
            align_str = ' '.join(map(str, hypo['alignment']))
            if args.unk_replace_dict != '':
                hypo_str = replace_unk(hypo_str, align_str, orig, dataset.dst_dict.unk_string())
            print('H{}\t{}\t{}'.format(id_str, hypo['score'], hypo_str))
            print('A{}\t{}'.format(id_str, align_str))

    if args.interactive:
        for line in sys.stdin:
            tokens = tokenizer.Tokenizer.tokenize(line, dataset.src_dict, add_if_not_exist=False).long()
            start = dataset.src_dict.pad() + 1
            positions = torch.arange(start, start + len(tokens)).type_as(tokens)
            if use_cuda:
                positions = positions.cuda()
                tokens = tokens.cuda()
            translations = translator.generate(Variable(tokens.view(1, -1)), Variable(positions.view(1, -1)))
            hypos = translations[0]
            display_hypotheses(None, tokens, line, None, hypos[:min(len(hypos), args.nbest)])

    else:
        def maybe_remove_bpe(tokens):
            """Helper for removing BPE symbols from a hypothesis."""
            if args.remove_bpe is None:
                return tokens
            assert (tokens == dataset.dst_dict.pad()).sum() == 0
            hypo_minus_bpe = dataset.dst_dict.string(tokens, args.remove_bpe)
            return tokenizer.Tokenizer.tokenize(hypo_minus_bpe, dataset.dst_dict, add_if_not_exist=True)

        # Generate and compute BLEU score
        scorer = bleu.Scorer(dataset.dst_dict.pad(), dataset.dst_dict.eos(), dataset.dst_dict.unk())
        itr = dataset.dataloader(args.gen_subset, batch_size=args.batch_size,
                                 max_positions=args.max_positions,
                                 skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test)
        num_sentences = 0
        with progress_bar(itr, smoothing=0, leave=False) as t:
            wps_meter = TimeMeter()
            gen_timer = StopwatchMeter()
            translations = translator.generate_batched_itr(
                t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                cuda_device=0 if use_cuda else None, timer=gen_timer)
            for id, src, ref, hypos in translations:
                ref = ref.int().cpu()
                top_hypo = hypos[0]['tokens'].int().cpu()
                scorer.add(maybe_remove_bpe(ref), maybe_remove_bpe(top_hypo))
                display_hypotheses(id, src, None, ref, hypos[:min(len(hypos), args.nbest)])

                wps_meter.update(src.size(0))
                t.set_postfix(wps='{:5d}'.format(round(wps_meter.avg)), refresh=False)
                num_sentences += 1

        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} tokens/s)'.format(
            num_sentences, gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))


if __name__ == '__main__':
    main()
