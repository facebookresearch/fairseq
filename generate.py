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

from fairseq import bleu, data, options, utils, tokenizer
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.progress_bar import progress_bar
from fairseq.sequence_generator import SequenceGenerator


def main():
    parser = options.get_parser('Generation')
    parser.add_argument('--path', metavar='FILE', required=True, default='./checkpoint_best.pt',
                        help='path to model file')
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
    print('| loading model from {}'.format(args.path))
    model, dataset = utils.load_checkpoint_for_inference(args.path, args.data)

    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    if not args.interactive:
        print('| {} {} {} examples'.format(args.data, args.gen_subset, len(dataset.splits[args.gen_subset])))

    # Optimize model for generation
    model.make_generation_fast_(args.beam, not args.no_beamable_mm)

    # Initialize generator
    translator = SequenceGenerator(model, dataset.dst_dict, beam_size=args.beam,
                                   stop_early=(not args.no_early_stop),
                                   normalize_scores=(not args.unnormalized),
                                   len_penalty=args.lenpen)
    if use_cuda:
        translator.cuda()

    bpe_symbol = '@@ ' if args.remove_bpe else None
    def display_hypotheses(id, src, orig, ref, hypos):
        id_str = '' if id is None else '-{}'.format(id)
        print('S{}\t{}'.format(id_str, to_sentence(dataset.src_dict, src, bpe_symbol)))
        if orig is not None:
            print('O{}\t{}'.format(id_str, orig.strip()))
        if ref is not None:
            print('T{}\t{}'.format(id_str, to_sentence(dataset.dst_dict, ref, bpe_symbol, ref_unk=True)))
        for hypo in hypos:
            print('H{}\t{}\t{}'.format(
                id_str, hypo['score'], to_sentence(dataset.dst_dict, hypo['tokens'], bpe_symbol)))
            print('A{}\t{}'.format(id_str, ' '.join(map(str, hypo['alignment']))))

    if args.interactive:
        print('> ', end='', flush=True)
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
            print('> ', end='', flush=True)

    else:
        non_bpe_dict = {}
        def maybe_remove_bpe_and_reindex(tokens):
            """Helper for removing BPE symbols from a tensor of indices.

            If BPE removal is enabled, the returned tensor is reindexed
            using a new dictionary that is created on-the-fly."""
            if not args.remove_bpe:
                return tokens
            assert (tokens == dataset.dst_dict.pad()).sum() == 0
            return torch.IntTensor([
                non_bpe_dict.setdefault(w, len(non_bpe_dict))
                for w in to_sentence(dataset.dst_dict, tokens, bpe_symbol).split(' ')
            ])

        # Generate and compute BLEU score
        scorer = bleu.Scorer(
            dataset.dst_dict.pad() if not args.remove_bpe else -1,
            dataset.dst_dict.eos() if not args.remove_bpe else -1,
            dataset.dst_dict.unk())
        itr = dataset.dataloader(args.gen_subset, batch_size=args.batch_size)
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
                scorer.add(maybe_remove_bpe_and_reindex(ref), maybe_remove_bpe_and_reindex(top_hypo))
                display_hypotheses(id, src, None, ref, hypos[:min(len(hypos), args.nbest)])

                wps_meter.update(src.size(0))
                t.set_postfix(wps='{:5d}'.format(round(wps_meter.avg)))
                num_sentences += 1

        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} tokens/s)'.format(
            num_sentences, gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))


def to_token(dict, i, runk):
    return runk if i == dict.unk() else dict[i]


def to_sentence(dict, tokens, bpe_symbol=None, ref_unk=False):
    if torch.is_tensor(tokens) and tokens.dim() == 2:
        sentences = [to_sentence(dict, token) for token in tokens]
        return '\n'.join(sentences)
    eos = dict.eos()
    unk = dict[dict.unk()]
    runk = '<{}>'.format(unk) if ref_unk else unk
    sent = ' '.join([to_token(dict, i, runk) for i in tokens if i != eos])
    if bpe_symbol is not None:
        sent = sent.replace(bpe_symbol, '')
    return sent


if __name__ == '__main__':
    main()
