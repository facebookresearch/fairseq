# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import inspect
from typing import Any, Dict, List
from torch.autograd import Variable
import fairseq
from fairseq import metrics, utils, data
from fairseq.criterions import FairseqCriterion
#from fairseq.sequence_generator import SequenceGenerator


class FairseqSequenceCriterion(FairseqCriterion):
    """Base class for sequence-level criterions."""

    def __init__(self, task, seq_hypos_dropout, seq_unkpen,
                 seq_sampling, seq_max_len_a, seq_max_len_b,
                 seq_beam, seq_scorer):
        super().__init__(task)

        self.dst_dict = task.tgt_dict
        self.pad_idx = task.tgt_dict.pad()
        self.eos_idx = task.tgt_dict.eos()
        self.unk_idx = task.tgt_dict.unk()

        self.seq_hypos_dropout = seq_hypos_dropout
        self.seq_unkpen = seq_unkpen
        self.seq_sampling = seq_sampling
        self.seq_max_len_a = seq_max_len_a
        self.seq_max_len_b = seq_max_len_b
        self.seq_beam = seq_beam
        self.seq_scorer = seq_scorer
        
        self._generator = None
        self._scorer = None

    #
    # Methods to be defined in sequence-level criterions
    #

    def prepare_sample_and_hypotheses(self, model, sample, hypos):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        return sample, hypos

    def sequence_forward(self, net_output, model, sample):
        """Compute the sequence-level loss for the given hypotheses.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError

    @staticmethod
    def grad_denom(sample_sizes):
        """Compute the gradient denominator for a set of sample sizes."""
        return sum(sample_sizes)

    #
    # Helper methods
    #

    def add_reference_to_hypotheses(self, sample, hypos):
        """Add the reference translation to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """
        if 'includes_reference' in sample:
            return hypos
        sample['includes_reference'] = True

        target = sample['target'].data
        for i, hypos_i in enumerate(hypos):
            # insert reference as first hypothesis
            #ref = utils.lstrip_pad(target[i, :], self.pad_idx)
            ref = utils.strip_pad(target[i, :], self.pad_idx)
            hypos_i.insert(0, {
                'tokens': ref,
                'score': None,
            })
        return hypos

    def add_bleu_to_hypotheses(self, sample, hypos):
        """Add BLEU scores to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """
        if 'includes_bleu' in sample:
            return hypos
        sample['includes_bleu'] = True

        if self._scorer is None:
            self.create_sequence_scorer()

        target = sample['target'].data.int()
        for i, hypos_i in enumerate(hypos):
            #ref = utils.lstrip_pad(target[i, :], self.pad_idx).cpu()
            ref = utils.strip_pad(target[i, :], self.pad_idx).cpu()
            #r = self.dst_dict.string(ref, bpe_symbol='@@ ', escape_unk=True)
            #r = fairseq.tokenizer.Tokenizer.tokenize(r, self.dst_dict, add_if_not_exist=True)
            #r = fairseq.tokenizer.tokenize_line(r)
            for hypo in hypos_i:
                #h = self.dst_dict.string(hypo['tokens'].int().cpu(), bpe_symbol='@@ ')
                #h = fairseq.tokenizer.Tokenizer.tokenize(h, self.dst_dict, add_if_not_exist=True)
                #h = fairseq.tokenizer.tokenize_line(h)
                # use +1 smoothing for sentence BLEU
                self._scorer.add(ref, hypo['tokens'].int().cpu())
                hypo['bleu'] = self._scorer.score()
        return hypos

    def create_sequence_scorer(self):
        if self.seq_scorer == "bleu":
            config = fairseq.scoring.bleu.BleuConfig(pad=self.pad_idx,
                                                     eos=self.eos_idx,
                                                     unk=self.unk_idx)
            self._scorer = fairseq.scoring.bleu.Scorer(config)
        else:
            raise Exception("Unknown sequence scorer {}".format(self.seq_scorer))

    def get_hypothesis_scores(self, net_output, sample, score_pad=False):
        """Return a tensor of model scores for each hypothesis.

        The returned tensor has dimensions [bsz, nhypos, hypolen]. This can be
        called from sequence_forward.
        """
        bsz, nhypos, hypolen, _ = net_output.size()
        hypotheses = Variable(sample['hypotheses'], requires_grad=False).view(bsz, nhypos, hypolen, 1)
        scores = net_output.gather(3, hypotheses)
        if not score_pad:
            scores = scores * hypotheses.ne(self.pad_idx).float()
        return scores.squeeze(3)

    def get_hypothesis_lengths(self, net_output, sample):
        """Return a tensor of hypothesis lengths.

        The returned tensor has dimensions [bsz, nhypos]. This can be called
        from sequence_forward.
        """
        bsz, nhypos, hypolen, _ = net_output.size()
        lengths = sample['hypotheses'].view(bsz, nhypos, hypolen).ne(self.pad_idx).sum(2).float()
        return Variable(lengths, requires_grad=False)

    #
    # Methods required by FairseqCriterion
    #

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model_state = model.training
        if model_state:
            model.train(self.seq_hypos_dropout)

        # generate hypotheses
        hypos = self._generate_hypotheses(model, sample)

        model.train(model_state)

        # apply any criterion-specific modifications to the sample/hypotheses
        sample, hypos = self.prepare_sample_and_hypotheses(model, sample, hypos)

        # create a new sample out of the hypotheses
        sample = self._update_sample_with_hypos(sample, hypos)


        # run forward and get sequence-level loss
        net_output = self.get_net_output(model, sample)
        loss, sample_size, logging_output = self.sequence_forward(net_output, model, sample)

        return loss, sample_size, logging_output

    def get_net_output(self, model, sample):
        """Return model outputs as log probabilities."""
        net_output = model(**sample['net_input'])
        return F.log_softmax(net_output, dim=1).view(
            sample['bsz'], sample['num_hypos_per_batch'], -1, net_output.size(1))


    def _generate_hypotheses(self, model, sample):
        # initialize generator
        if self._generator is None:
            self._generator = fairseq.sequence_generator.SequenceGenerator(
#                [model], self.dst_dict, unk_penalty=self.seq_unkpen, sampling=self.seq_sampling)
              [model], self.dst_dict, unk_penalty=self.seq_unkpen)
            self._generator.cuda()

        # generate hypotheses
        # input = sample['net_input']

        # print(input)
        
        # srclen = input['src_tokens'].size(1)
        # hypos = self._generator.generate(
        #     input['src_tokens'], input['src_positions'],
        #     maxlen=int(self.seq_max_len_a * srclen + self.seq_max_len_b),
        #     beam_size=self.seq_beam)

        # # add reference to the set of hypotheses
        # if self.args.seq_keep_reference:
        #     hypos = self.add_reference_to_hypotheses(sample, hypos)

        hypos = self._generator(sample)

        return hypos

    def _update_sample_with_hypos(self, sample, hypos):
        num_hypos_per_batch = len(hypos[0])
        assert all(len(h) == num_hypos_per_batch for h in hypos)

        # TODO(noa): the below needs to be adapted to the new collate methods
        
        def repeat_num_hypos_times(t):
            return t.repeat(1, num_hypos_per_batch).view(num_hypos_per_batch*t.size(0), t.size(1))

        input = sample['net_input']
        bsz = input['src_tokens'].size(0)
        input['src_tokens'].data = repeat_num_hypos_times(input['src_tokens'].data)
        input['src_lengths'].data = repeat_num_hypos_times(input['src_lengths'].data)

        input_hypos = [h['tokens'] for hypos_i in hypos for h in hypos_i]
        sample['hypotheses'] = data.LanguagePairDataset.collate_tokens(
            input_hypos, self.pad_idx, self.eos_idx, left_pad=True, move_eos_to_beginning=False)
        input['input_tokens'].data = data.LanguagePairDataset.collate_tokens(
            input_hypos, self.pad_idx, self.eos_idx, left_pad=True, move_eos_to_beginning=True)
        input['input_lengths'].data = data.LanguagePairDataset.collate_positions(
            input_hypos, self.pad_idx, left_pad=True)

        sample['target'].data = repeat_num_hypos_times(sample['target'].data)
        sample['ntokens'] = sample['target'].data.ne(self.pad_idx).sum()
        sample['bsz'] = bsz
        sample['num_hypos_per_batch'] = num_hypos_per_batch
        return sample


class BleuScorer(object):

    def __init__(self, pad, eos, unk):
        self._scorer = bleu.Scorer(pad, eos, unk)

    def score(self, ref, hypo):
        self._scorer.reset(one_init=True)
        self._scorer.add(ref, hypo)
        return self._scorer.score()
