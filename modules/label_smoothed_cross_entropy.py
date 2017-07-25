import torch
from torch.autograd.variable import Variable

class LabelSmoothedCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, target, eps, padding_idx, weights):
        grad_input = input.new(input.size()).zero_()
        target = target.view(target.size(0), 1)
        grad_input = grad_input.scatter_(grad_input.dim() - 1, target, eps - 1)

        norm =  grad_input.size(-1)
        if weights is not None:
            norm = weights.sum()
            grad_input.mul(weights.view(1, weights.size(0)).expand_as(grad_input))

        if padding_idx is not None:
            norm -= 1 if weights is None else weights[padding_idx]
            grad_input.select(grad_input.dim() - 1, padding_idx).fill_(0)

        grad_input = grad_input.add(-eps / norm)

        ctx.grad_input = grad_input
        return input.new([grad_input.view(-1).dot(input.view(-1))])

    @staticmethod
    def backward(ctx, grad):
        return Variable(ctx.grad_input, volatile=True) * grad, None, None, None, None


def label_smoothed_cross_entropy(input, target, eps=0.1, padding_idx=None, weights=None):
    return LabelSmoothedCrossEntropy.apply(input, target, eps, padding_idx, weights)

