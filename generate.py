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

from fairseq import bleu, options, utils, tokenizer
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

    # Load model and dataset
    print('| loading model(s) from {}'.format(', '.join(args.path)))
    models, dataset = utils.load_ensemble_for_inference(args.path, args.data, args.gen_subset)

    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    if not args.interactive:
        print('| {} {} {} examples'.format(args.data, args.gen_subset, len(dataset.splits[args.gen_subset])))

    # Optimize model for generation
    for model in models:
        model.make_generation_fast_(not args.no_beamable_mm)

    # Initialize generator
    translator = SequenceGenerator(models, dataset.dst_dict, beam_size=args.beam,
                                   stop_early=(not args.no_early_stop),
                                   normalize_scores=(not args.unnormalized),
                                   len_penalty=args.lenpen)
    align_dict = {}
    if args.unk_replace_dict != '':
        assert args.interactive, "Unkown words replacing requires access to original source and is only" \
                                 "supported in interactive mode"
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

    if use_cuda:
        translator.cuda()

    bpe_symbol = '@@ ' if args.remove_bpe else None
    def display_hypotheses(id, src, orig, ref, hypos):
        if args.quiet:
            return
        id_str = '' if id is None else '-{}'.format(id)
        src_str = to_sentence(dataset.src_dict, src, bpe_symbol)
        print('S{}\t{}'.format(id_str, src_str))
        if orig is not None:
            print('O{}\t{}'.format(id_str, orig.strip()))
        if ref is not None:
            print('T{}\t{}'.format(id_str, to_sentence(dataset.dst_dict, ref, bpe_symbol, ref_unk=True)))
        for hypo in hypos:
            hypo_str = to_sentence(dataset.dst_dict, hypo['tokens'], bpe_symbol)
            align_str = ' '.join(map(str, hypo['alignment']))
            if args.unk_replace_dict != '':
                hypo_str = replace_unk(hypo_str, align_str, orig, unk_symbol(dataset.dst_dict))
            print('H{}\t{}\t{}'.format(
                id_str, hypo['score'], hypo_str))
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
            if not args.remove_bpe:
                return tokens
            assert (tokens == dataset.dst_dict.pad()).sum() == 0
            hypo_minus_bpe = to_sentence(dataset.dst_dict, tokens, bpe_symbol)
            return tokenizer.Tokenizer.tokenize(hypo_minus_bpe, dataset.dst_dict, add_if_not_exist=True)

        # Generate and compute BLEU score
        scorer = bleu.Scorer(dataset.dst_dict.pad(), dataset.dst_dict.eos(), dataset.dst_dict.unk())
        itr = dataset.dataloader(args.gen_subset, batch_size=args.batch_size, max_positions=args.max_positions)
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


def to_token(dict, i, runk):
    return runk if i == dict.unk() else dict[i]


def unk_symbol(dict, ref_unk=False):
    return '<{}>'.format(dict.unk_word) if ref_unk else dict.unk_word


def to_sentence(dict, tokens, bpe_symbol=None, ref_unk=False):
    if torch.is_tensor(tokens) and tokens.dim() == 2:
        sentences = [to_sentence(dict, token) for token in tokens]
        return '\n'.join(sentences)
    eos = dict.eos()
    runk = unk_symbol(dict, ref_unk=ref_unk)
    sent = ' '.join([to_token(dict, i, runk) for i in tokens if i != eos])
    if bpe_symbol is not None:
        sent = sent.replace(bpe_symbol, '')
    return sent


if __name__ == '__main__':
    main()
