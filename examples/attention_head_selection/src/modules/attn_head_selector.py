# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math


class AttnHeadSelector(nn.Module):
    """
    Latent variable modeling of attention head selection
    """
    def __init__(
        self, num_tasks, num_layers,
        total_num_heads, num_heads,
        select_strategy="group",
        head_select_temp=5.0
    ):
        super(AttnHeadSelector, self).__init__()
        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.total_num_heads = total_num_heads
        self.num_heads = num_heads
        self.select_strategy = select_strategy
        self.temp = head_select_temp

        self.head_logits = torch.nn.Parameter(
            torch.Tensor(self.num_tasks, self.num_layers, total_num_heads),
            requires_grad=True
        )
        nn.init.uniform_(
            self.head_logits, a=math.log(0.01),
            b=math.log(1.0)
        )

    def gumbel_sample(self, logits, tau=1.0):
        gumbels1 = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels2 = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
        return y_soft

    def subset_select(self, y_soft, topk, dim=-1):
        top_values, top_inds = torch.topk(y_soft, k=topk, dim=dim)
        top_ret = 1.0 - top_values.detach() + top_values
        return top_inds.detach(), top_ret

    def group_selet(self, y_soft, topk, dim=-1):
        # top_values: (num_tasks, num_layers, topk)
        top_values, top_inds = torch.max(
            y_soft.view(self.num_tasks, self.num_layers, -1, topk), dim=2
        )
        top_inds = top_inds * topk + torch.arange(topk, device=top_inds.device).unsqueeze(0).unsqueeze(1)
        top_ret = 1.0 - top_values.detach() + top_values
        return top_inds.detach(), top_ret

    def head_select(self, task_ids=None):
        # gumbel_sample
        self.head_samples = self.gumbel_sample(self.head_logits, tau=self.temp)
        # head select
        if self.select_strategy == "subset":
            self.subset_heads, self.subset_weights = self.subset_select(
                self.head_samples,
                topk=self.num_heads,
            )
        elif self.select_strategy == "group":
            self.subset_heads, self.subset_weights = self.group_selet(
                self.head_samples,
                topk=self.num_heads,
            )
        else:
            raise ValueError("{} is not supported".format(self.select_strategy))

        self.batch_subset = self.subset_heads[task_ids, :, :]
        self.batch_weights = self.subset_weights[task_ids, :, :]

    def forward(self, layer_idx):
        assert layer_idx is not None
        batch_subset = self.batch_subset[:, layer_idx, :]
        batch_weights = self.batch_weights[:, layer_idx, :]
        return batch_subset, batch_weights
