# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from torch import nn

from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('composite_loss')
class CompositeLoss(FairseqCriterion):
    """This is a composite loss that, given a list of model outputs and a list of targets,
    computes an average of losses for each output-target pair"""

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--underlying-criterion', type=str, metavar='VAL', required=True,
                            help='underlying criterion to use for the composite loss')
        # fmt: on

    @staticmethod
    def build_underlying_criterion(args, task):
        saved_criterion = args.criterion
        args.criterion = args.underlying_criterion
        assert saved_criterion != args.underlying_criterion
        underlying_criterion = task.build_criterion(args)
        args.criterion = saved_criterion
        return underlying_criterion

    @classmethod
    def build_criterion(cls, args, task):
        underlying_criterion = CompositeLoss.build_underlying_criterion(args, task)

        class FakeModel(nn.Module):

            def __init__(self, model, net_out, target):
                super().__init__()
                self.model = model
                self.net_out = net_out
                self.target = target

            def forward(self, **unused):
                return self.net_out

            def get_normalized_probs(self, net_output, log_probs, sample=None):
                return self.model.get_normalized_probs(net_output, log_probs, sample=sample)

            def get_targets(self, *unused):
                return self.target

            @property
            def decoder(self):
                return self.model.decoder

        class _CompositeLoss(FairseqCriterion):

            def __init__(self, args, task, underlying_criterion):
                super().__init__(args, task)
                self.underlying_criterion = underlying_criterion

            def forward(self, model, sample, reduce=True):
                net_outputs = model(**sample['net_input'])
                targets = sample['target']

                bsz = targets[0].size(0)
                loss = net_outputs[0][0].new(1 if reduce else bsz).float().zero_()

                sample_size = 0
                logging_output = {}
                for o, t in zip(net_outputs[0], targets):
                    m = FakeModel(model, (o, net_outputs[1]), t)
                    sample['target'] = t
                    l, ss, logging_output = self.underlying_criterion(m, sample, reduce)
                    loss += l
                    sample_size += ss

                loss.div_(len(targets))
                sample_size /= len(targets)

                logging_output['loss'] = utils.item(loss.data) if reduce else loss.data
                return loss, sample_size, logging_output

            @staticmethod
            def aggregate_logging_outputs(logging_outputs):
                return underlying_criterion.__class__.aggregate_logging_outputs(logging_outputs)

        return _CompositeLoss(args, task, underlying_criterion)
