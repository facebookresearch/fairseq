#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from fairseq import utils
from fairseq.data import encoders
from fairseq.models import BaseFairseqModel


def from_pretrained(
    model_name_or_path,
    checkpoint_file='model.pt',
    data_name_or_path='.',
    archive_map=None,
    **kwargs
):
    from fairseq import checkpoint_utils, file_utils

    if archive_map is not None:
        if model_name_or_path in archive_map:
            model_name_or_path = archive_map[model_name_or_path]
        if data_name_or_path is not None and data_name_or_path in archive_map:
            data_name_or_path = archive_map[data_name_or_path]

    model_path = file_utils.load_archive_file(model_name_or_path)

    # convenience hack for loading data and BPE codes from model archive
    if data_name_or_path.startswith('.'):
        kwargs['data'] = os.path.abspath(os.path.join(model_path, data_name_or_path))
    else:
        kwargs['data'] = file_utils.load_archive_file(data_name_or_path)
    for file, arg in {
        'code': 'bpe_codes',
        'bpecodes': 'bpe_codes',
        'sentencepiece.bpe.model': 'sentencepiece_vocab',
    }.items():
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            kwargs[arg] = path

    models, args, task = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(model_path, cpt) for cpt in checkpoint_file.split(':')],
        arg_overrides=kwargs,
    )

    return {
        'args': args,
        'task': task,
        'models': models,
    }


class Generator(BaseFairseqModel):
    """PyTorch Hub API for generating sequences from a pre-trained translation
    or language model."""

    def __init__(self, args, task, models):
        self.args = args
        self.task = task
        self.models = models
        self.src_dict = task.source_dictionary
        self.tgt_dict = task.target_dictionary
        self.use_cuda = torch.cuda.is_available() and not getattr(args, 'cpu', False)

        if self.use_cuda:
            if getattr(args, 'fp16', False):
                self.half()
            self.cuda()

        # optimize model for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=(
                    None if getattr(args, 'no_beamable_mm', False)
                    else getattr(args, 'beam', 5)
                ),
                need_attn=getattr(args, 'print_alignment', False),
            )

        self.generator = self.task.build_generator(args)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(getattr(args, 'replace_unk', None))

        self.tokenizer = encoders.build_tokenizer(args)
        self.bpe = encoders.build_bpe(args)

    def generate(self, src_str, verbose=False):

        def preprocess(s):
            if self.tokenizer is not None:
                s = self.tokenizer.encode(s)
            if self.bpe is not None:
                s = self.bpe.encode(s)
            return s

        def postprocess(s):
            if self.bpe is not None:
                s = self.bpe.decode(s)
            if self.tokenizer is not None:
                s = self.tokenizer.decode(s)
            return s

        src_str = preprocess(src_str)
        tokens = self.src_dict.encode_line(src_str, add_if_not_exist=False).long()
        if verbose:
            src_str_with_unk = self.src_dict.string(tokens)
            print('S\t{}'.format(src_str_with_unk))

        dataset = self.task.build_dataset_for_inference([tokens], [tokens.numel()])
        sample = dataset.collater([dataset[0]])
        if self.use_cuda:
            sample = utils.move_to_cuda(sample)

        translations = self.task.inference_step(self.generator, self.models, sample)

        # Process top predictions
        for hypo in translations[0][:min(len(translations), getattr(self.args, 'nbest', 1))]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=self.align_dict,
                tgt_dict=self.tgt_dict,
            )
            hypo_str = postprocess(hypo_str)
            if verbose:
                print('H\t{}\t{}'.format(hypo['score'], hypo_str))
                print('P\t{}'.format(
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
                if getattr(self.args, 'print_alignment', False):
                    print('A\t{}'.format(
                        ' '.join(map(lambda x: str(utils.item(x)), alignment))
                    ))

        return hypo_str
