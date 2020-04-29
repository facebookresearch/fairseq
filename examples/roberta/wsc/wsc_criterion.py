# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import encoders
from fairseq.criterions import LegacyFairseqCriterion, register_criterion


@register_criterion('wsc')
class WSCCriterion(LegacyFairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        if self.args.save_predictions is not None:
            self.prediction_h = open(self.args.save_predictions, 'w')
        else:
            self.prediction_h = None
        self.bpe = encoders.build_bpe(args)
        self.tokenizer = encoders.build_tokenizer(args)

    def __del__(self):
        if self.prediction_h is not None:
            self.prediction_h.close()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--wsc-margin-alpha', type=float, metavar='A', default=1.0)
        parser.add_argument('--wsc-margin-beta', type=float, metavar='B', default=0.0)
        parser.add_argument('--wsc-cross-entropy', action='store_true',
                            help='use cross entropy formulation instead of margin loss')
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')

    def get_masked_input(self, tokens, mask):
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.task.mask
        return masked_tokens

    def get_lprobs(self, model, tokens, mask):
        logits, _ = model(src_tokens=self.get_masked_input(tokens, mask))
        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float)
        scores = lprobs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
        mask = mask.type_as(scores)
        scores = (scores * mask).sum(dim=-1) / mask.sum(dim=-1)
        return scores

    def get_loss(self, query_lprobs, cand_lprobs):
        if self.args.wsc_cross_entropy:
            return F.cross_entropy(
                torch.cat([query_lprobs, cand_lprobs]).unsqueeze(0),
                query_lprobs.new([0]).long(),
            )
        else:
            return (
                - query_lprobs
                + self.args.wsc_margin_alpha * (
                    cand_lprobs - query_lprobs + self.args.wsc_margin_beta
                ).clamp(min=0)
            ).sum()

    def forward(self, model, sample, reduce=True):
        # compute loss and accuracy
        loss, nloss = 0., 0
        ncorrect, nqueries = 0, 0

        for i, label in enumerate(sample['labels']):
            query_lprobs = self.get_lprobs(
                model,
                sample['query_tokens'][i].unsqueeze(0),
                sample['query_masks'][i].unsqueeze(0),
            )
            cand_lprobs = self.get_lprobs(
                model,
                sample['candidate_tokens'][i],
                sample['candidate_masks'][i],
            )

            pred = (query_lprobs >= cand_lprobs).all().item()

            if label is not None:
                label = 1 if label else 0
                ncorrect += 1 if pred == label else 0
                nqueries += 1

            if label:
                # only compute a loss for positive instances
                nloss += 1
                loss += self.get_loss(query_lprobs, cand_lprobs)

            id = sample['id'][i].item()
            if self.prediction_h is not None:
                print('{}\t{}\t{}'.format(id, pred, label), file=self.prediction_h)

        if nloss == 0:
            loss = torch.tensor(0.0, requires_grad=True)

        sample_size = nqueries if nqueries > 0 else 1
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'ncorrect': ncorrect,
            'nqueries': nqueries,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
        nqueries = sum(log.get('nqueries', 0) for log in logging_outputs)
        if nqueries > 0:
            agg_output['accuracy'] = ncorrect / float(nqueries)

        return agg_output


@register_criterion('winogrande')
class WinograndeCriterion(WSCCriterion):
    def forward(self, model, sample, reduce=True):
        # compute loss and accuracy
        query_lprobs = self.get_lprobs(
            model,
            sample['query_tokens'],
            sample['query_masks'],
        )
        cand_lprobs = self.get_lprobs(
            model,
            sample['candidate_tokens'],
            sample['candidate_masks'],
        )
        pred = query_lprobs >= cand_lprobs
        loss = self.get_loss(query_lprobs, cand_lprobs)

        sample_size = sample['query_tokens'].size(0)
        ncorrect = pred.sum().item()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'ncorrect': ncorrect,
            'nqueries': sample_size,
        }
        return loss, sample_size, logging_output
