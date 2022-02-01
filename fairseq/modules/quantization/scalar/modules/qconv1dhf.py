import torch
import torch.nn as nn

from fairseq.modules.quantization.scalar.ops import emulate_int

# For IntConv1DHF, we take the same structure as the traditional Conv1D
# but add elements for quantization from fairseq


class IntConv1DHF(nn.Module):
    """
    Quantized counterpart of the Conv1D module that applies QuantNoise during training.
    The Conv1D is a 1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.

    Args:
        - nf (int): The number of output features.
        - nx (int): The number of input features.
        - bias: bias or not
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-through estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick.
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(
        self,
        nf,
        nx,
        bias=True,
        p=0,
        update_step=3000,
        bits=8,
        method="histogram",
    ):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.chosen_bias = bias
        if self.chosen_bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.register_parameter("bias", None)

        # quantization parameters
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def forward(self, x):
        # train with QuantNoise and evaluate the fully quantized network
        p = self.p if self.training else 1

        # update parameters every 100 iterations
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1

        # quantize weight
        weight_quantized, self.scale, self.zero_point = emulate_int(
            self.weight.detach(),
            bits=self.bits,
            method=self.method,
            scale=self.scale,
            zero_point=self.zero_point,
        )

        # mask to apply noise
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)

        # using straight-through estimator (STE)
        clamp_low = -self.scale * self.zero_point
        clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
        weight = (
            torch.clamp(self.weight, clamp_low.item(), clamp_high.item())
            + noise.detach()
        )

        # The forward for Conv1D is different from the Linear one
        # https://github.com/huggingface/transformers/src/transformers/modeling_utils.py#L1764

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(
            self.bias, x.view(-1, x.size(-1)), weight
        )  # weight, not self.weight
        x = x.view(*size_out)
        return x

    def extra_repr(self):
        return "nf={}, bias={}, quant_noise={}, bits={}, method={}".format(
            self.nf,
            self.bias is not None,
            self.p,
            self.bits,
            self.method,
        )


# End
