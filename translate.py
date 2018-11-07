#!/usr/bin/env python  
# -*- coding:utf-8 _*-
# @author:Chao Bei 
# @file: translate.py 
# @time: 2018/11/07
# @software: PyCharm 

from collections import namedtuple
import numpy as np
import torch

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator


class Translate(object):
    def __init__(self, args):
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1
        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'
        self.args = args
        print(args)
        self.Batch = namedtuple('Batch', 'srcs tokens lengths')
        self.Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')
        # Setup task, e.g., translation
        self.task = tasks.setup_task(args)
        self.use_cuda = torch.cuda.is_available() and not args.cpu
        # Load ensemble
        print('| loading model(s) from {}'.format(args.path))
        model_paths = args.path.split(':')
        models, model_args = utils.load_ensemble_for_inference(model_paths, self.task,
                                                               model_arg_overrides=eval(args.model_overrides))

        # Set dictionaries
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()

        # Initialize generator
        self.translator = SequenceGenerator(
            models, self.tgt_dict, beam_size=args.beam, minlen=args.min_len,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
            diverse_beam_groups=args.diverse_beam_groups, diverse_beam_strength=args.diverse_beam_strength,
        )

        if self.use_cuda:
            self.translator.cuda()

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in models]
        )
        pass

    def translate(self, inputs):
        indices = []
        results = []
        for batch, batch_indices in self._make_batches(inputs, self.args, self.task, self.max_positions):
            indices.extend(batch_indices)
            results += self._process_batch(batch)

        for i in np.argsort(indices):
            result = results[i]
            print(result.src_str)
            for hypo, pos_scores, align in zip(result.hypos, result.pos_scores, result.alignments):
                print(hypo)
                print(pos_scores)
                if align is not None:
                    print(align)

    def _make_result(self, src_str, hypos):
        result = self.Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            pos_scores=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), self.args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=self.align_dict,
                tgt_dict=self.tgt_dict,
                remove_bpe=self.args.remove_bpe,
            )
            result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.pos_scores.append('P\t{}'.format(
                ' '.join(map(
                    lambda x: '{:.4f}'.format(x),
                    hypo['positional_scores'].tolist(),
                ))
            ))
            result.alignments.append(
                'A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment)))
                if self.args.print_alignment else None
            )
        return result

    def _process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths

        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        encoder_input = {'src_tokens': tokens, 'src_lengths': lengths}
        translations = self.translator.generate(
            encoder_input,
            maxlen=int(self.args.max_len_a * tokens.size(1) + self.args.max_len_b),
        )

        return [self._make_result(batch.srcs[i], t) for i, t in enumerate(translations)]

    def _make_batches(self, lines, args, task, max_positions):
        tokens = [
            tokenizer.Tokenizer.tokenize(src_str, task.source_dictionary, add_if_not_exist=False).long()
            for src_str in lines
        ]
        lengths = np.array([t.numel() for t in tokens])
        itr = task.get_batch_iterator(
            dataset=data.LanguagePairDataset(tokens, lengths, task.source_dictionary),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=max_positions,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            yield self.Batch(
                srcs=[lines[i] for i in batch['id']],
                tokens=batch['net_input']['src_tokens'],
                lengths=batch['net_input']['src_lengths'],
            ), batch['id']
