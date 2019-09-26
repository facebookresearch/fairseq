#!/usr/bin/env python
# encoding: utf-8
# File Name: AUCPRHingeLoss.py
# Author: Jiezhong Qiu
# Create Time: 2019/08/28 14:28
# TODO:

import torch
import torch.nn as nn
import numpy as np

class AUCPRHingeLoss(nn.Module):
    """area under the precision-recall curve loss,
    Reference: "Scalable Learning of Non-Decomposable Objectives".
    Args:
        num_classes: number of output classes in you model
        num_anchors: number of points to approximate to the precision-recall
            curve. We calculate the area under precision-recall cure based on
            Riemann sum over anchors. More anchors will provide better
            approximation but longer training time
        precision_range_(lower, upper): define the area of precision-recall
            curve belongs to [precision_range_lower, precision_range_upper]
    """

    def __init__(
        self,
        num_classes,
        num_anchors,
        precision_range_lower,
        precision_range_upper,
        weights=None
    ):
        super(AUCPRHingeLoss, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.precision_range = (precision_range_lower, precision_range_upper)

        # Create precision anchor values and distance between anchors.
        self.precision_values, self.delta = _prec_anchors_and_delta(
            self.precision_range, self.num_anchors
        )

        # notation is [b_k] in paper, Parameter of shape [C, K]
        # where `C = number of classes` `K = num_anchors`
        self.biases = nn.Parameter(
            torch.zeros(num_classes, num_anchors)
        )
        self.lambdas = nn.Parameter(
            torch.ones(num_classes, num_anchors)
        )

    def forward(
        self,
        logits,
        targets,
        reduce=True,
        size_average=True,
        weights=None
    ):
        #number of classes
        num_classes = 1 if logits.dim() == 1 else logits.size(1)

        if self.num_classes != num_classes:
            raise ValueError(
                "num classes is %d while logits width is \
                    %d" % (self.num_classes, num_classes)
            )

        labels, weights = _onehot_labels_weights(
            logits, targets, weights=weights
        )

        # Lagrange multipliers
        # Lagrange multipliers are updated based on reverse direction of gradient
        # 1D `Tensor` of shape [num_anchors]
        lambdas = lagrange_multiplier(self.lambdas)

        # Shape [batch_size, num_classes, num_anchors]
        self.precision_values = self.precision_values.type(logits.type())
        hinge_loss = _weighted_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=1.0 + lambdas * (1.0 - self.precision_values),
            negative_weights=lambdas * self.precision_values,
        )

        # shape [num_classes]
        class_priors = _maybe_create_label_priors(labels, weights=weights)

        # lambda_term: [num_classes, num_anchors]
        lambda_term = class_priors.unsqueeze(-1) * (
            lambdas * (1.0 - self.precision_values)
        )

        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term

        # Riemann sum over anchors, and normalized by precision range
        # loss: Tensor[batch_size, num_anchors]
        per_label_loss = per_anchor_loss.sum(2) * self.delta

        # Normalize the AUC such that a perfect score function will have AUC 1.0.
        # Because precision_range is discretized into num_anchors + 1 intervals
        # but only num_anchors terms are included in the Riemann sum, the
        # effective length of the integration interval is `delta` less than the
        # length of precision_range.
        loss = per_label_loss / \
            (self.precision_range[1] - self.precision_range[0] - self.delta)

        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()


def _onehot_labels_weights(logits, targets, weights=None):

    batch_size, num_classes = logits.size()
    # Converts targets to one-hot representation: batch_size * num_classes
    labels = torch.FloatTensor(batch_size, num_classes).zero_().type(targets.type())
    labels = labels.scatter(1, targets.unsqueeze(1).data, 1)

    if weights is None:
        weights = torch.FloatTensor(batch_size).fill_(1.0)

    if weights.dim() == 1:
        weights.unsqueeze_(-1)

    return labels.type(logits.type()), weights.type(logits.type())


def _prec_anchors_and_delta(precision_range, num_anchors):
    # Validate precision_range.
    if len(precision_range) != 2:
        raise ValueError(
            "length of precision_range (%d) must be 2" % len(precision_range)
        )
    if not 0 <= precision_range[0] <= precision_range[1] <= 1:
        raise ValueError(
            "precision values must follow 0 <= %f <= %f <= 1"
            % (precision_range[0], precision_range[1])
        )

    # Sets precision_values uniformly between min_precision and max_precision.
    precision_values = np.linspace(  # this is one of the hacky part
        start=precision_range[0], stop=precision_range[1], num=num_anchors + 2
    )[1:-1]

    delta = precision_values[0] - precision_range[0]
    return torch.FloatTensor(precision_values), delta


def _maybe_create_label_priors(
    labels,
    class_priors=None,
    weights=None,
    positive_pseudocount=1.0,
    negative_pseudocount=1.0,
):
    """Creates an op to maintain and update label prior probabilities.
     For each label, the label priors are estimated as
         (P + sum_i w_i y_i) / (P + N + sum_i w_i),
     where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
     positive labels, and N is a pseudo-count of negative labels. The index i
     ranges over all labels observed during all evaluations of the returned op.
    """
    if class_priors is not None:
        return class_priors

    weighted_label_counts = (weights * labels).sum(0)

    weight_sum = weights.sum(0)

    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )

    return class_priors


def _weighted_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):
    positive_weights_is_tensor = torch.is_tensor(positive_weights)
    negative_weights_is_tensor = torch.is_tensor(negative_weights)

    # Validate positive_weights and negative_weights
    if positive_weights_is_tensor ^ negative_weights_is_tensor:
        raise ValueError(
            "positive_weights and negative_weights must be same shape Tensor "
            "or both be scalars. But positive_weight_is_tensor: %r, while "
            "negative_weight_is_tensor: %r"
            % (positive_weights_is_tensor, negative_weights_is_tensor)
        )

    if positive_weights_is_tensor and (
        positive_weights.size() != negative_weights.size()
    ):
        raise ValueError(
            "shape of positive_weights and negative_weights "
            "must be the same! "
            "shape of positive_weights is {0}, "
            "but shape of negative_weights is {1}"
            % (positive_weights.size(), negative_weights.size())
        )

    # positive_term: Tensor [N, C] or [N, C, K]
    positive_term = (1 - logits).clamp(min=0) * labels
    negative_term = (1 + logits).clamp(min=0) * (1 - labels)

    if positive_weights_is_tensor and positive_term.dim() == 2:
        return (
            positive_term.unsqueeze(-1) * positive_weights
            + negative_term.unsqueeze(-1) * negative_weights
        )
    else:
        return positive_term * positive_weights + negative_term * negative_weights


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)
