#!/usr/bin/env python
# encoding: utf-8
"""
@author: Wang Qiang
@contact: wangqiangneu@gmail.com
@desc: connection schema between layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DynamicLinearCombination(nn.Module):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """
    def __init__(self, args, is_encoder, include_sublayer=False):
        super(DynamicLinearCombination, self).__init__()
        self.normalize_learned_weight = args.normalize_learned_weight
        self.normalized_weight = None
        self.weight_type = args.weight_type
        self.out_dropout = args.history_dropout
        self.normalize_before = args.encoder_normalize_before if is_encoder else args.decoder_normalize_before
        self.dim = args.encoder_embed_dim if is_encoder else args.decoder_embed_dim

        # transformer encoder has 2 sub-layers, decoder has 3 sub-layers
        if include_sublayer:
            layer_num = 1 + (2 * args.encoder_layers if is_encoder else 3 * args.decoder_layers)
        else:
            layer_num = 1 + (args.encoder_layers if is_encoder else args.decoder_layers)

        # init weights and corresponding masks
        learnable = args.encoder_learnable if is_encoder else args.decoder_learnable
        self.weight, self.weight_mask = self._init(layer_num, args.init_value, args.weight_type,
                                                   args.history_window_size, learnable)

        # init triangular layer norm
        if args.normalize_embed:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(layer_num)])
        else:
            self.layer_norms = nn.ModuleList([nn.Sequential()] + [nn.LayerNorm(self.dim) for _ in range(layer_num-1)])

        # states
        self.count = 0
        self.layers = []

    @staticmethod
    def _init_mask(n_layer, window_size):
        mask = np.zeros([n_layer, n_layer], dtype=np.float32)
        # all preceding layers
        if window_size == -1:
            for i in range(mask.shape[0]):
                mask[i, :(i+1)] = 1
        else:
            for i in range(mask.shape[0]):
                mask[i, max(0, i + 1 - window_size): (i+1)] = 1
        return torch.from_numpy(mask)

    @staticmethod
    def _init_weight(np_mask, dim=1, init_value='avg', learnable=True):
        np_weight = np.copy(np_mask)
        if init_value == 'avg':
            np_weight = np_weight / np.sum(np_weight, axis=1, keepdims=True)
        elif init_value == 'one':
            np_weight[:, :] = 1.
        else:
            raise ValueError('unknown init_value:{}'.format(init_value))
        weight_tensor = torch.from_numpy(np_weight).unsqueeze(2)
        if dim > 1:
            weight_tensor = weight_tensor.repeat(1, 1, dim)
        weight_tensor = torch.nn.Parameter(weight_tensor, requires_grad=learnable)
        return weight_tensor

    def _init(self, layer_num, init_value, weight_type, window_size=-1, learnable=True):
        """
            weight shape is:
             1. L x L for weight type='scalar'
             2. L x L x G for weight type='group_scalar'
             3. L x L x H for weight type='vector'
             mask shape is L x L
            :return:
        """
        # L x L
        mask_tensor = self._init_mask(layer_num, window_size)
        if weight_type == 'scalar':
            self.last_dim = 1
        elif weight_type == 'vector':
            self.last_dim = self.dim
        elif weight_type.startswith('scalar_'):
            n = int(weight_type.split('_')[1])
            assert self.dim % n == 0
            self.last_dim = n
        else:
            raise ValueError('unknown weight_type:{}'.format(weight_type))
        weight_tensor = self._init_weight(mask_tensor.numpy(), self.last_dim, init_value,
                                          learnable=learnable)
        return weight_tensor, mask_tensor

    def push(self, layer):
        self.count += 1

        # first layer
        if self.count == 1:
            self.layers.append(self.layer_norms[0](layer))
            # compatible when running on CPU
            if layer.is_cuda and not self.weight_mask.is_cuda:
                self.weight_mask = self.weight_mask.cuda()
            if self.normalize_learned_weight:
                weight = self.weight.masked_fill((self.weight_mask == 0).unsqueeze(2), float('-inf'))
                self.normalized_weight = F.softmax(weight, dim=1)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count-1](layer)

        self.layers.append(layer)

    def _pick_weights(self):
        weight = self.normalized_weight if self.normalize_learned_weight else self.weight
        weight = weight[self.count - 1, : self.count, :].view(-1, 1, 1, self.last_dim)
        return weight

    def pop(self):
        assert len(self.layers) > 0

        # D x 1 x 1 x [1, H/G, H]
        weights = self._pick_weights()
        # D x T x B x H
        layers = torch.stack(self.layers, 0)
        # linear combination
        if self.weight_type in ['scalar', 'vector']:
            ret = (layers * weights).sum(0)
        else:
            D, T, B, H = layers.size()
            layers = layers.view(D, T, B, -1, weights.size(-1))
            weights = weights.unsqueeze(3)
            ret = (layers * weights).sum(0).view(T, B, H)

        if self.normalize_before:
            if self.out_dropout > 0:
                return F.dropout(ret, p=self.out_dropout, training=self.training)
            else:
                return ret
        if self.out_dropout > 0:
            return F.dropout(self.layer_norms[self.count-1](ret), p=self.out_dropout, training=self.training)
        else:
            return self.layer_norms[self.count-1](ret)

    def clean(self):
        self.count = 0
        self.layers = []

    def forward(self):
        pass



