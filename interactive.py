#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import sys
import torch
from torch.autograd import Variable

from fairseq import options, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator


def main(args):
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load ensemble
    print('| loading model(s) from {}'.format(', '.join(args.path)))
    models, model_args = utils.load_ensemble_for_inference(args.path, data_dir=args.data)
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
        unk_penalty=args.unkpen)
    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    print('| Type the input sentence and press return:')
    for src_str in sys.stdin:
        src_str = src_str.strip()
        src_tokens = tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
        if use_cuda:
            src_tokens = src_tokens.cuda()
        src_lengths = src_tokens.new([src_tokens.numel()])
        translations = translator.generate(
            Variable(src_tokens.view(1, -1)),
            Variable(src_lengths.view(-1)),
        )
        hypos = translations[0]
        print('O\t{}'.format(src_str))

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
            print('H\t{}\t{}'.format(hypo['score'], hypo_str))
            print('A\t{}'.format(' '.join(map(str, alignment))))


if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = parser.parse_args()
    main(args)
