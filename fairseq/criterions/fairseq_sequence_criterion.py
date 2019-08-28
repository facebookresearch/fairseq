# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from torch.autograd import Variable
import torch.nn.functional as F

from fairseq import bleu, data, sequence_generator, tokenizer, utils
from . import FairseqCriterion

class FairseqSequenceCriterion(FairseqCriterion):
    """Base class for sequence-level criterions."""

    def __init__(self, args, task):#dst_dict):
        super().__init__(args, task)
        self.args = args
        self.task = task
        self.eos_idx = task.target_dictionary.eos() if task.target_dictionary is not None else -100
        self.unk_idx = task.target_dictionary.unk() if task.target_dictionary is not None else -100

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

    @staticmethod
    def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning):
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            if left_pad:
                copy_tensor(v, res[i][size-len(v):])
            else:
                copy_tensor(v, res[i][:len(v)])
        return res

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
            ref = utils.strip_pad(target[i, :], self.padding_idx)
            hypos_i.insert(0, {
                'tokens': ref,
                'score': None,
            })
        return hypos

    def add_bleu_to_hypotheses(self, sample, hypos):
        """Add BLEU scores to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """
        tgt_dict = self.task.target_dictionary
        if 'includes_bleu' in sample:
            return hypos
        sample['includes_bleu'] = True

        if self._scorer is None:
            self.create_sequence_scorer()

        #target = sample['target'].data.int()
        for i, hypos_i in enumerate(hypos):
            target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
            target_str = tgt_dict.string(target_tokens, '@@ ', escape_unk=True)
            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
            # r = self.task.target_dictionary.string(ref, bpe_symbol='@@ ', escape_unk=True)
            # r = tokenizer.Tokenizer.tokenize(r, self.task.target_dictionary, add_if_not_exist=True)
            for hypo in hypos_i:
                hypo_str = tgt_dict.string(hypo['tokens'].int().cpu(), '@@ ', escape_unk=True)
                hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
                # h = self.task.target_dictionary.string(hypo['tokens'].int().cpu(), bpe_symbol='@@ ')
                # h = tokenizer.Tokenizer.tokenize(h, self.task.target_dictionary, add_if_not_exist=True)
                # use +1 smoothing for sentence BLEU
                hypo['bleu'] = self._scorer.score(target_tokens, hypo_tokens)
        return hypos

    def create_sequence_scorer(self):
        if self.args.seq_scorer == "bleu":
            self._scorer = BleuScorer(self.padding_idx, self.eos_idx, self.unk_idx)
        else:
            raise Exception("Unknown sequence scorer {}".format(self.args.seq_scorer))

    def get_hypothesis_scores(self, net_output, sample, score_pad=False):
        """Return a tensor of model scores for each hypothesis.

        The returned tensor has dimensions [bsz, nhypos, hypolen]. This can be
        called from sequence_forward.
        """
        bsz, nhypos, hypolen, _ = net_output.size()
        hypotheses = Variable(sample['hypotheses'], requires_grad=False).view(bsz, nhypos, hypolen, 1)
        scores = net_output.gather(3, hypotheses)
        if not score_pad:
            scores = scores * hypotheses.ne(self.padding_idx).float()
        return scores.squeeze(3)

    def get_hypothesis_lengths(self, net_output, sample):
        """Return a tensor of hypothesis lengths.

        The returned tensor has dimensions [bsz, nhypos]. This can be called
        from sequence_forward.
        """
        bsz, nhypos, hypolen, _ = net_output.size()
        lengths = sample['hypotheses'].view(bsz, nhypos, hypolen).ne(self.padding_idx).sum(2).float()
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
            model.train(self.args.seq_hypos_dropout)

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
        decoder_out = net_output[0].view(-1, net_output[0].size(-1))
        return F.log_softmax(decoder_out, dim=1).view(
            sample['bsz'], sample['num_hypos_per_batch'], -1, decoder_out.size(1))


    def _generate_hypotheses(self, model, sample):
        # initialize generator
        if self._generator is None:
            gen_args = utils.get_training_generator_args(self.args)
            self._generator = self.task.build_generator(gen_args)

        # generate hypotheses
        prefix_tokens = None
        if self.args.seq_prefix_size > 0:
            prefix_tokens = sample['target'][:, :self.args.seq_prefix_size]
        hypos = self.task.inference_step(self._generator, [model], sample, prefix_tokens)
        # add reference to the set of hypotheses
        if self.args.seq_keep_reference:
            hypos = self.add_reference_to_hypotheses(sample, hypos)

        return hypos

    def _update_sample_with_hypos(self, sample, hypos):
        num_hypos_per_batch = len(hypos[0])
        assert all(len(h) == num_hypos_per_batch for h in hypos)

        def repeat_num_hypos_times(t):
            return t.repeat(1, num_hypos_per_batch).view(num_hypos_per_batch*t.size(0), t.size(1))

        input = sample['net_input']
        bsz = input['src_tokens'].size(0)
        input['src_tokens'].data = repeat_num_hypos_times(input['src_tokens'].data)
        input['src_lengths'].data = repeat_num_hypos_times(input['src_lengths'].data.view(input['src_lengths'].size(0), -1)).squeeze()

        input_hypos = [h['tokens'] for hypos_i in hypos for h in hypos_i]
        sample['hypotheses'] = self.collate_tokens(
            input_hypos, self.padding_idx, self.eos_idx, True, False)
        input['prev_output_tokens'] = self.collate_tokens(
            input_hypos, self.padding_idx, self.eos_idx, True, True)
        #input['prev_output_tokens'].data = repeat_num_hypos_times(input['prev_output_tokens'].data)

        sample['target'].data = repeat_num_hypos_times(sample['target'].data)
        sample['ntokens'] = sample['target'].data.ne(self.padding_idx).sum()
        sample['nsentences'] = num_hypos_per_batch*sample['nsentences']
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

