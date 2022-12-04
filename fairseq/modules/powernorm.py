#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : MaskPowerNorm.py
# Distributed under MIT License.

import torch
import torch.nn as nn

__all__ = ['MaskPowerNorm']


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


class GroupScaling1D(nn.Module):
    r"""Scales inputs by the second moment for the entire layer.
    """

    def __init__(self, eps=1e-5, group_num=4):
        super(GroupScaling1D, self).__init__()
        self.eps = eps
        self.group_num = group_num

    def extra_repr(self):
        return f'eps={self.eps}, group={self.group_num}'

    def forward(self, input):
        # calculate second moment
        # different group use different mean
        T, B, C = input.shape[0], input.shape[1], input.shape[2]
        Cg = C // self.group_num
        gn_input = input.contiguous().reshape(T, B, self.group_num, Cg)
        moment2 = torch.repeat_interleave(torch.mean(gn_input * gn_input, dim=3, keepdim=True), \
                                          repeats=Cg, dim=-1).contiguous().reshape(T, B, C)
        # divide out second moment
        return input / torch.sqrt(moment2 + self.eps)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class PowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_phi, eps, afwd, abkw, ema_gz, \
                debug, warmup_iters, current_iter, mask_x):
        ctx.eps = eps
        ctx.debug = debug
        current_iter = current_iter.item()
        ctx.current_iter = current_iter
        ctx.warmup_iters = warmup_iters
        ctx.abkw = abkw
        N, C, H, W = x.size()
        x2 = (mask_x * mask_x).mean(dim=0)

        var = x2.reshape(1, C, 1, 1)
        if current_iter <= warmup_iters:
            z = x / (var + eps).sqrt()
        else:
            z = x / (running_phi + eps).sqrt()

        y = z
        ctx.save_for_backward(z, var, weight, ema_gz)

        if current_iter < warmup_iters:
            running_phi.copy_(
                running_phi * (current_iter - 1) / current_iter + var.mean(dim=0, keepdim=True) / current_iter)
        running_phi.copy_(afwd * running_phi + (1 - afwd) * var.mean(dim=0, keepdim=True))
        y = weight.reshape(1, C, 1, 1) * y + bias.reshape(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        abkw = ctx.abkw

        N, C, H, W = grad_output.size()
        z, var, weight, ema_gz = ctx.saved_variables

        y = z
        g = grad_output * weight.reshape(1, C, 1, 1)
        g = g * 1

        approx_grad_g = (g - (1 - abkw) * ema_gz * z)
        ema_gz.add_((approx_grad_g * z).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True))

        gx = 1. / torch.sqrt(var + eps) * approx_grad_g
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), \
               None, None, None, None, None, None, None, None, None, None


class MaskPowerNorm(nn.Module):
    """
    An implementation of masked batch normalization, used for testing the numerical
    stability.
    """

    def __init__(self, num_features, eps=1e-5, alpha_fwd=0.9, alpha_bkw=0.9, \
                 affine=True, warmup_iters=10000, group_num=1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        self.register_parameter('weight', nn.Parameter(torch.ones(num_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_features)))
        self.register_buffer('running_phi', torch.ones(1, num_features, 1, 1))
        self.register_buffer('ema_gz', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('iters', torch.zeros(1).type(torch.LongTensor))

        self.afwd = alpha_fwd
        self.abkw = alpha_bkw

        self.eps = eps
        self.debug = False
        self.warmup_iters = warmup_iters
        self.gp = GroupScaling1D(group_num=group_num)
        self.group_num = group_num

    def extra_repr(self):
        return '{num_features}, eps={eps}, alpha_fwd={afwd}, alpha_bkw={abkw}, ' \
               'affine={affine}, warmup={warmup_iters}, group_num={group_num}'.format(**self.__dict__)

    def forward(self, input, pad_mask=None, is_encoder=False):
        """
        input:  T x B x C -> B x C x T
             :  B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        """
        shaped_input = (len(input.shape) == 2)
        if shaped_input:
            input = input.unsqueeze(0)
        input = self.gp(input)

        # construct the mask_input, size to be (BxL) x C: L is the real length here
        if pad_mask is None:
            mask_input = input.clone()
        else:
            # Transpose the bn_mask (B x T -> T x B)
            bn_mask = ~pad_mask
            bn_mask = bn_mask.transpose(0, 1)
            mask_input = input[bn_mask, :]

        mask_input = mask_input.reshape(-1, self.num_features)

        input = input.permute(1, 2, 0).contiguous()
        input_shape = input.size()
        input = input.reshape(input.size(0), self.num_features, -1)
        input = input.unsqueeze(-1)

        if self.training:
            self.iters.copy_(self.iters + 1)
            output = PowerFunction.apply(input, self.weight, self.bias, self.running_phi, self.eps, \
                                         self.afwd, self.abkw, self.ema_gz, self.debug, self.warmup_iters, self.iters,
                                         mask_input)

        else:
            N, C, H, W = input.size()
            var = self.running_phi
            output = input / (var + self.eps).sqrt()
            output = self.weight.reshape(1, C, 1, 1) * output + self.bias.reshape(1, C, 1, 1)

        output = output.reshape(input_shape)
        output = output.permute(2, 0, 1).contiguous()
        # Reshape it.
        if shaped_input:
            output = output.squeeze(0)

        return output

