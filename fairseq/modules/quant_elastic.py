# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math

from fairseq.modules.quant_noise import quant_noise

WEIGHT_QUANT_METHOD_CHOICES = ["bwn"]
ACT_QUANT_METHOD_CHOICES = ["bwn", "elastic"]


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class ElasticQuantBinarizerSigned(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(
                input, num_bits, symmetric=True, init_method="default"
            )
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, "alpha = {:.6f} becomes non-positive".format(alpha)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = (
                ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
            )
        else:
            grad_alpha = (
                (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                .sum()
                .unsqueeze(dim=0)
            )
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class ElasticQuantBinarizerUnsigned(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(
                input, num_bits, symmetric=False, init_method="default"
            )
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, "alpha = {:.6f} becomes non-positive".format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        if num_bits != 1:
            w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = (
            (
                (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle * (-q_w + q_w.round())
                )
                * grad_output
                * grad_scale
            )
            .sum()
            .unsqueeze(dim=0)
        )
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class AlphaInit(nn.Parameter):
    def __init__(self, tensor):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, "already initialized."
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method="default"):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == "default":
            init_val = (
                2 * tensor.abs().mean() / math.sqrt(Qp)
                if symmetric
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
            )
        elif init_method == "uniform":
            init_val = 1.0 / (2 * Qp + 1) if symmetric else 1.0 / Qp

        self._initialize(init_val)


class BwnQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
    Ref: https://arxiv.org/abs/1603.05279
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            e = input.mean()
            result = (input - e).sign().mul(m.expand(s))
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            e = input.mean()
            result = (input - e).sign().mul(m.expand(s))

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def act_quant_fn(input, clip_val, num_bits, symmetric, quant_method, layerwise):
    if num_bits == 32:
        return input
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    elif quant_method == "elastic" and num_bits >= 1:
        quant_fn = (
            ElasticQuantBinarizerSigned if symmetric else ElasticQuantBinarizerUnsigned
        )
    else:
        raise ValueError("Unknown quant_method")

    input = quant_fn.apply(input, clip_val, num_bits, layerwise)

    return input


def weight_quant_fn(weight, clip_val, num_bits, symmetric, quant_method, layerwise):
    if num_bits == 32:
        return weight
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    else:
        raise ValueError(f"Unknown {quant_method=}")

    weight = quant_fn.apply(weight, clip_val, num_bits, layerwise)
    return weight


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *args,
        clip_val=2.5,
        weight_bits=8,
        input_bits=8,
        learnable=False,
        symmetric=True,
        weight_layerwise=True,
        input_layerwise=True,
        weight_quant_method="bwn",
        input_quant_method="elastic",
        transpose=False,
        **kwargs,
    ):
        super(QuantizeLinear, self).__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self._build_weight_clip_val(weight_quant_method, learnable, init_val=clip_val)
        self._build_input_clip_val(input_quant_method, learnable, init_val=clip_val)
        self.transpose = transpose
        self.move = LearnableBias(
            self.weight.shape[0] if transpose else self.weight.shape[1]
        )

    def _build_weight_clip_val(self, quant_method, learnable, init_val):
        if quant_method == "uniform":
            self.register_buffer("weight_clip_val", torch.tensor([-init_val, init_val]))
            if learnable:
                self.weight_clip_val = nn.Parameter(self.weight_clip_val)
        elif quant_method == "elastic":
            assert learnable, "Elastic method must use learnable step size!"
            self.weight_clip_val = AlphaInit(
                torch.tensor(1.0)
            )  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer("weight_clip_val", None)

    def _build_input_clip_val(self, quant_method, learnable, init_val):
        if quant_method == "uniform":
            self.register_buffer("input_clip_val", torch.tensor([-init_val, init_val]))
            if learnable:
                self.input_clip_val = nn.Parameter(self.input_clip_val)
        elif quant_method == "elastic" or quant_method == "bwn":
            assert learnable, "Elastic method must use learnable step size!"
            self.input_clip_val = AlphaInit(
                torch.tensor(1.0)
            )  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer("input_clip_val", None)

    def forward(self, input):
        # quantize weight
        weight = weight_quant_fn(
            self.weight.t() if self.transpose else self.weight,
            self.weight_clip_val,
            num_bits=self.weight_bits,
            symmetric=self.symmetric,
            quant_method=self.weight_quant_method,
            layerwise=self.weight_layerwise,
        )
        # quantize input
        input = self.move(input)
        input = act_quant_fn(
            input,
            self.input_clip_val,
            num_bits=self.input_bits,
            symmetric=self.symmetric,
            quant_method=self.input_quant_method,
            layerwise=self.input_layerwise,
        )
        out = nn.functional.linear(input, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(
        self,
        *args,
        clip_val=2.5,
        weight_bits=8,
        learnable=False,
        symmetric=True,
        embed_layerwise=False,
        weight_quant_method="bwn",
        **kwargs,
    ):
        super(QuantizeEmbedding, self).__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.embed_layerwise = embed_layerwise
        self.weight_quant_method = weight_quant_method
        self._build_embed_clip_val(weight_quant_method, learnable, init_val=clip_val)

    def _build_embed_clip_val(self, quant_method, learnable, init_val):
        if quant_method == "uniform":
            self.register_buffer("embed_clip_val", torch.tensor([-init_val, init_val]))
            if learnable:
                self.embed_clip_val = nn.Parameter(self.embed_clip_val)
        elif quant_method == "elastic":
            assert learnable, "Elastic method must use learnable step size!"
            self.embed_clip_val = AlphaInit(
                torch.tensor(1.0)
            )  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer("embed_clip_val", None)

    def forward(self, input):
        weight = weight_quant_fn(
            self.weight,
            self.embed_clip_val,
            num_bits=self.weight_bits,
            symmetric=self.symmetric,
            quant_method=self.weight_quant_method,
            layerwise=self.embed_layerwise,
        )

        out = nn.functional.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return out


class QuantizeElasticMixin:
    def _maybe_build_quantize_linear(
        self, input_dim, output_dim, bias=True, transpose=False
    ):
        if self.weight_bits == 32:
            layer = quant_noise(
                nn.Linear(input_dim, output_dim, bias=bias),
                self.q_noise,
                self.qn_block_size,
            )
        else:
            layer = QuantizeLinear(
                input_dim,
                output_dim,
                bias=bias,
                weight_bits=self.weight_bits,
                weight_quant_method=self.weight_quant_method,
                learnable=self.learnable_scaling,
                symmetric=self.symmetric_quant,
                transpose=transpose,
            )
        return layer
