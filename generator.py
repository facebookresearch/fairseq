#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import namedtuple

import torch
import html
import os
from sacremoses import MosesTokenizer, MosesDetokenizer
from subword_nmt import apply_bpe

from fairseq import checkpoint_utils, options, tasks, utils, file_utils

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')


class Generator(object):

    def __init__(self, task, models, args, src_bpe=None, bpe_symbol='@@ '):
        self.task = task
        self.models = models
        self.src_dict = task.source_dictionary
        self.tgt_dict = task.target_dictionary
        self.src_bpe = src_bpe
        self.use_cuda = torch.cuda.is_available() and not args.cpu
        self.args = args

        self.args.remove_bpe = bpe_symbol

        # optimize model for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        self.generator = self.task.build_generator(args)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in models]
        )

        if hasattr(args, 'source_lang'):
            self.tokenizer = MosesTokenizer(lang=args.source_lang)
        else:
            self.tokenizer = MosesTokenizer()

        if src_bpe is not None:
            bpe_parser = apply_bpe.create_parser()
            bpe_args = bpe_parser.parse_args(['--codes', self.src_bpe])
            self.bpe = apply_bpe.BPE(bpe_args.codes, bpe_args.merges, bpe_args.separator, None, bpe_args.glossaries)
        else:
            self.bpe = None

    def generate(self, src_str, verbose=False):

        src_str = self.tokenizer.tokenize(src_str, return_str=True)
        if self.bpe:
            src_str = self.bpe.process_line(src_str)

        for batch in self.make_batches([src_str], self.args, self.task, self.max_positions):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            src_tokens = utils.strip_pad(src_tokens, self.tgt_dict.pad())

        if self.src_dict is not None:
            src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
            if verbose:
                print('S\t{}'.format(src_str))

        # Process top predictions
        for hypo in translations[0][:min(len(translations), self.args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=self.align_dict,
                tgt_dict=self.tgt_dict,
                remove_bpe=self.args.remove_bpe,
            )
            if verbose:
                print('H\t{}\t{}'.format(hypo['score'], hypo_str))
                print('P\t{}'.format(
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
                if self.args.print_alignment:
                    print('A\t{}'.format(
                        ' '.join(map(lambda x: str(utils.item(x)), alignment))
                    ))

        return html.unescape(hypo_str)

    @classmethod
    def from_pretrained(cls, parser, *args, model_name_or_path, data_name_or_path, **kwargs):
        model_path = file_utils.load_archive_file(model_name_or_path)
        data_path = file_utils.load_archive_file(data_name_or_path)
        checkpoint_path = os.path.join(model_path, 'model.pt')

        task_name = kwargs.get('task', 'translation')

        # set data and parse
        model_args = options.parse_args_and_arch(parser, input_args=[data_path, '--task', task_name])

        # override any kwargs passed in
        if kwargs is not None:
            for arg_name, arg_val in kwargs.items():
                setattr(model_args, arg_name, arg_val)

        utils.import_user_module(args)

        if model_args.buffer_size < 1:
            model_args.buffer_size = 1
        if model_args.max_tokens is None and model_args.max_sentences is None:
            model_args.max_sentences = 1

        assert not model_args.sampling or model_args.nbest == model_args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not model_args.max_sentences or model_args.max_sentences <= model_args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        print(model_args)

        task = tasks.setup_task(model_args)
        print("loading model checkpoint from {}".format(checkpoint_path))

        model, _model_args = checkpoint_utils.load_model_ensemble([checkpoint_path], task=task)
        src_bpe = os.path.join(model_path, 'bpecodes')
        if not os.path.exists(src_bpe):
            src_bpe = None

        return cls(task, model, model_args, src_bpe, kwargs.get('remove_bpe', '@@ '))

    def make_batches(self, lines, args, task, max_positions):
        tokens = [
            task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
            for src_str in lines
        ]
        lengths = torch.LongTensor([t.numel() for t in tokens])
        itr = task.get_batch_iterator(
            dataset=task.build_dataset_for_inference(tokens, lengths),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=max_positions,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            yield Batch(
                ids=batch['id'],
                src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
            )
