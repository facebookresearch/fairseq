# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
from torch.nn.modules.loss import _Loss


class LatentLayersKLLoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, layer_samples, lang_idx, update_num, sample_size):
        prior = self.args.prior
        samples = layer_samples[lang_idx]
        eps = 1e-7
        if prior == "uniform":
            # uniform prior
            kl_loss = (samples * (
                torch.log(samples + eps) - math.log(0.5)
            )).sum(-1)
        elif prior == "agged_posterior":
            # aggregated posterior
            y_t = torch.stack([x.detach() for x in layer_samples], dim=0)
            agged_q = torch.sum(y_t, dim=0)
            row_norm = agged_q.sum(-1)
            normed_agg_q = agged_q / row_norm
            kl_loss = (samples * (
                torch.log(samples + eps) - torch.log(normed_agg_q + eps))).sum(-1)
        else:
            raise NotImplementedError("The specified prior is not implemented.")

        # normalized by number of layers
        kl_loss /= layer_samples[0].size()[0]
        kl_weight = min(
            self.args.sparsity_weight,
            (update_num - self.args.soft_update) * self.args.sparsity_weight / self.args.anneal_updates
        )
        kl_loss *= kl_weight * sample_size
        return kl_loss


class LatentLayersSparsityLoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def is_valid(self, update_num):
        if self.args.target_layers <= 0:
            return False
        return update_num > (self.args.soft_update + self.args.anneal_updates)

    def forward(self, layer_samples_list, update_num, sample_size):
        batch_loss = 0
        share_loss = 0
        global_sparsity_loss = 0
        layer_samples = torch.stack(layer_samples_list, dim=0)
        if ((self.args.target_layers > 0 or self.args.share_weight > 0) and
                update_num > (self.args.soft_update + self.args.anneal_updates)):
            # anneal sparsity weight
            if update_num < (self.args.anneal_updates + self.args.soft_update):
                weight_anneal = 0
            elif update_num < (2 * self.args.anneal_updates + self.args.soft_update):
                weight_anneal = (
                    (update_num - self.args.soft_update - self.args.anneal_updates)
                    * self.args.share_weight / self.args.anneal_updates
                )
            else:
                weight_anneal = 1
            # compute ratio among languages
            layer_utilization = torch.sum(layer_samples, dim=0)
            layer_utilization /= layer_samples.size()[0]
            if self.args.share_weight > 0:
                # encouraging sharing across languages
                share_loss = sum(-1.0 * v * math.log(v) for v in layer_utilization if v > 0)
                batch_loss += weight_anneal * self.args.share_weight * sample_size * share_loss
            if self.args.target_layers > 0:
                # computed expected number of layers selected
                expeted_layers = sum(layer_utilization)
                # compute l2 loss wrt target number of layers
                global_sparsity_loss = (expeted_layers - self.args.target_layers) ** 2
                batch_loss += weight_anneal * self.args.share_weight * sample_size * global_sparsity_loss
        return batch_loss
