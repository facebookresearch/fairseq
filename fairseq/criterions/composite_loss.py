# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from torch import nn

from fairseq import utils, options
from . import FairseqCriterion, register_criterion


@register_criterion('composite_loss')
class CompositeLoss(FairseqCriterion):
    """This is a composite loss that, given a list of model outputs and a list of targets,
    computes an average of losses for each output-target pair"""

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--underlying-criterion', type=str, metavar='VAL', required=True,
                            help='underlying criterion to use for the composite loss')
        parser.add_argument('--loss-weights', type=str, metavar='EXPR', default=None,
                            help='if set, provides 1 weight per target for each respective loss')

    def __init__(self, args, task):
        super().__init__(args, task)
        saved_criterion = args.criterion
        args.criterion = args.underlying_criterion

        assert saved_criterion != args.underlying_criterion

        self.underlying_criterion = task.build_criterion(args)
        args.criterion = saved_criterion
        self.weights = options.eval_str_list(args.loss_weights, type=float)

    class FakeModel(nn.Module):
        def __init__(self, model, net_out, target):
            super(CompositeLoss.FakeModel, self).__init__()
            self.model = model
            self.net_out = net_out
            self.target = target

        def forward(self, **unused):
            return self.net_out

        def get_targets(self, *unused):
            return self.target

        @property
        def decoder(self):
            return self.model.decoder

    def forward(self, model, sample, reduce=True):
        net_outputs = model(**sample['net_input'])
        targets = sample['target']

        bsz = targets[0].size(0)
        loss = net_outputs[0][0].new(1 if reduce else bsz).zero_()

        sample_size = 0
        logging_output = {}
        for i, (o, t) in enumerate(zip(net_outputs[0], targets)):
            m = CompositeLoss.FakeModel(model, (o, net_outputs[1]), t)
            l, ss, logging_output = self.underlying_criterion(m, sample, reduce)
            if self.weights is not None:
                l *= self.weights[i]
            loss += l
            sample_size += ss

        loss.div_(len(targets))
        sample_size /= len(targets)

        logging_output['loss'] = utils.item(loss.data) if reduce else loss.data
        return loss, sample_size, logging_output

    def _aggregate_logging_outputs(self, logging_outputs):
        return self.underlying_criterion._aggregate_logging_outputs(logging_outputs)
