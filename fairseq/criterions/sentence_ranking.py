# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('sentence_ranking')
class SentenceRankingCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        if self.args.save_predictions is not None:
            self.prediction_h = open(self.args.save_predictions, 'w')
        else:
            self.prediction_h = None

    def __del__(self):
        if self.prediction_h is not None:
            self.prediction_h.close()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        parser.add_argument('--ranking-head-name',
                            default='sentence_classification_head',
                            help='name of the ranking head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute ranking loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.args.ranking_head_name in model.classification_heads
        ), 'model must provide sentence ranking head for --criterion=sentence_ranking'

        scores = []
        for idx in range(self.args.num_classes):
            score, _ = model(
                **sample['net_input{idx}'.format(idx=idx+1)],
                classification_head_name=self.args.ranking_head_name,
            )
            scores.append(score)

        logits = torch.cat(scores, dim=1)
        sample_size = logits.size(0)

        if 'target' in sample:
            targets = model.get_targets(sample, [logits]).view(-1)
            loss = F.nll_loss(
                F.log_softmax(logits, dim=-1, dtype=torch.float32),
                targets,
                reduction='sum',
            )
        else:
            targets = None
            loss = torch.tensor(0.0, requires_grad=True)

        if self.prediction_h is not None:
            preds = logits.argmax(dim=1)
            for i, (id, pred) in enumerate(zip(sample['id'].tolist(), preds.tolist())):
                if targets is not None:
                    label = targets[i].item()
                    print('{}\t{}\t{}'.format(id, pred, label), file=self.prediction_h)
                else:
                    print('{}\t{}'.format(id, pred), file=self.prediction_h)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        if targets is not None:
            logging_output.update(
                ncorrect=(logits.max(dim=1)[1] == targets).sum().item()
            )
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

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect/nsentences)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
