# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

import numpy as np
import torch

from fairseq import bleu, utils
from fairseq.data import Dictionary, language_pair_dataset
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks import register_task, translation


class BleuScorer(object):

    key = 'bleu'

    def __init__(self, tgt_dict, bpe_symbol='@@ '):
        self.tgt_dict = tgt_dict
        self.bpe_symbol = bpe_symbol
        self.scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        # use a fresh Dictionary for scoring, so that we can add new elements
        self.scoring_dict = Dictionary()

    def preprocess_ref(self, ref):
        ref = self.tgt_dict.string(ref, bpe_symbol=self.bpe_symbol, escape_unk=True)
        return self.scoring_dict.encode_line(ref, add_if_not_exist=True)

    def preprocess_hypo(self, hypo):
        hypo = hypo['tokens']
        hypo = self.tgt_dict.string(hypo.int().cpu(), bpe_symbol=self.bpe_symbol)
        return self.scoring_dict.encode_line(hypo, add_if_not_exist=True)

    def get_cost(self, ref, hypo):
        self.scorer.reset(one_init=True)
        self.scorer.add(ref, hypo)
        return 1. - (self.scorer.score() / 100.)

    def postprocess_costs(self, costs):
        return costs


class NormalizedBleuScorer(BleuScorer):

    key = 'normalized_bleu'

    def postprocess_costs(self, costs):
        max_costs = costs.max(dim=1, keepdim=True)[0]
        min_costs = costs.min(dim=1, keepdim=True)[0]
        return (costs - min_costs) / (max_costs - min_costs)


@register_task('translation_struct')
class TranslationStructuredPredictionTask(translation.TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Compared to :class:`TranslationTask`, this version performs
    generation during training and computes sequence-level losses.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        translation.TranslationTask.add_args(parser)
        parser.add_argument('--seq-beam', default=5, type=int, metavar='N',
                            help='beam size for sequence training')
        parser.add_argument('--seq-keep-reference', default=False, action='store_true',
                            help='retain the reference in the list of hypos')
        parser.add_argument('--seq-scorer', default='bleu', metavar='SCORER',
                            choices=['bleu', 'normalized_bleu'],
                            help='optimization metric for sequence level training')

        parser.add_argument('--seq-gen-with-dropout', default=False, action='store_true',
                            help='use dropout to generate hypos')
        parser.add_argument('--seq-max-len-a', default=0, type=float, metavar='N',
                            help='generate sequences of maximum length ax + b, '
                                 'where x is the source length')
        parser.add_argument('--seq-max-len-b', default=200, type=int, metavar='N',
                            help='generate sequences of maximum length ax + b, '
                                 'where x is the source length')
        parser.add_argument('--seq-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE tokens before scoring')
        parser.add_argument('--seq-sampling', default=False, action='store_true',
                            help='use sampling instead of beam search')
        parser.add_argument('--seq-unkpen', default=0, type=float,
                            help='unknown word penalty to be used in seq generation')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self._generator = None
        self._scorers = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return super(TranslationStructuredPredictionTask, cls).setup_task(args, **kwargs)

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions
        criterion = criterions.build_criterion(args, self)
        assert isinstance(criterion, criterions.FairseqSequenceCriterion)
        return criterion

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        # control dropout during generation
        model.train(self.args.seq_gen_with_dropout)

        # generate hypotheses
        self._generate_hypotheses(model, sample)

        return super().train_step(
            sample=sample,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            ignore_grad=ignore_grad,
        )

    def valid_step(self, sample, model, criterion):
        model.eval()
        self._generate_hypotheses(model, sample)
        return super().valid_step(sample=sample, model=model, criterion=criterion)

    def _generate_hypotheses(self, model, sample):
        # initialize generator
        if self._generator is None:
            self._generator = SequenceGenerator(
                self.target_dictionary,
                beam_size=self.args.seq_beam,
                max_len_a=self.args.seq_max_len_a,
                max_len_b=self.args.seq_max_len_b,
                unk_penalty=self.args.seq_unkpen,
                sampling=self.args.seq_sampling,
            )

        # generate hypotheses
        sample['hypos'] = self._generator.generate(
            [model],
            sample,
        )

        # add reference to the set of hypotheses
        if self.args.seq_keep_reference:
            self.add_reference_to_hypotheses(sample)

    def add_reference_to_hypotheses_(self, sample):
        """
        Add the reference translation to the set of hypotheses. This can be
        called from the criterion's forward.
        """
        if 'includes_reference' in sample:
            return
        sample['includes_reference'] = True
        target = sample['target']
        pad_idx = self.target_dictionary.pad()
        for i, hypos_i in enumerate(sample['hypos']):
            # insert reference as first hypothesis
            ref = utils.strip_pad(target[i, :], pad_idx)
            hypos_i.insert(0, {
                'tokens': ref,
                'score': None,
            })

    def get_new_sample_for_hypotheses(self, orig_sample):
        """
        Extract hypotheses from *orig_sample* and return a new sample where the target
        """
        ids = orig_sample['id'].tolist()
        pad_idx = self.source_dictionary.pad()
        samples = [
            {
                'id': ids[i],
                'source': utils.strip_pad(orig_sample['net_input']['src_tokens'][i, :], pad_idx),
                'target': hypo['tokens'],
            }
            for i, hypos_i in enumerate(orig_sample['hypos'])
            for hypo in hypos_i
        ]
        return language_pair_dataset.collate(
            samples, pad_idx=pad_idx, eos_idx=self.source_dictionary.eos(),
            left_pad_source=self.args.left_pad_source, left_pad_target=self.args.left_pad_target,
        )

    def get_sequence_scorer(self, scorer):
        if scorer not in self._scorers:
            tgt_dict = self.target_dictionary
            if scorer == 'bleu':
                self._scorers[scorer] = BleuScorer(tgt_dict, bpe_symbol=self.args.seq_remove_bpe)
            elif scorer == 'normalized_bleu':
                self._scorers[scorer] = NormalizedBleuScorer(tgt_dict, bpe_symbol=self.args.seq_remove_bpe)
            else:
                raise ValueError('Unknown sequence scorer {}'.format(scorer))
        return self._scorers[scorer]

    def get_costs(self, sample, scorer=None):
        """Get costs for hypotheses using the specified *scorer*."""
        if scorer is None:
            scorer = self.get_sequence_scorer(self.args.seq_scorer)

        bsz = len(sample['hypos'])
        nhypos = len(sample['hypos'][0])
        target = sample['target'].int()
        pad_idx = self.target_dictionary.pad()

        costs = torch.zeros(bsz, nhypos).to(sample['target'].device)
        for i, hypos_i in enumerate(sample['hypos']):
            ref = utils.strip_pad(target[i, :], pad_idx).cpu()
            ref = scorer.preprocess_ref(ref)
            for j, hypo in enumerate(hypos_i):
                costs[i, j] = scorer.get_cost(ref, scorer.preprocess_hypo(hypo))
        return scorer.postprocess_costs(costs)
