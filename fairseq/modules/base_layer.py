# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import sys
import numpy as np
from fairseq import utils
from fairseq.distributed import utils as distributed_utils
from fairseq.modules.layer_norm import LayerNorm


class BaseLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        expert_centroids = torch.empty(self.num_workers, args.decoder_embed_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter(
            "expert_centroids", torch.nn.Parameter(expert_centroids)
        )
        self.expert_network = nn.Sequential(
            *([BaseSublayer(args) for _ in range(args.base_sublayers)])
        )
        self.expert_id = distributed_utils.get_data_parallel_rank()
        self.shuffle = args.base_shuffle

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        features = input_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad

        if self.shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            shuffle_input_splits = [0] * self.num_workers
            for i in range(shuffle_sort.size(0)):
                shuffle_input_splits[i % self.num_workers] += 1
            shuffle_input_splits = torch.tensor(
                shuffle_input_splits, device=features.device
            )
            shuffle_output_splits = All2All.apply(shuffle_input_splits)

            features = All2All.apply(
                features[shuffle_sort],
                shuffle_input_splits.tolist(),
                shuffle_output_splits.tolist(),
            )

        with torch.no_grad():
            # Compute similarity of each token to each expert, for routing
            token_expert_affinities = features.matmul(
                self.expert_centroids.transpose(0, 1)
            )

        # Compute which token goes to which expert
        sort_by_expert, input_splits, output_splits = (
            self.balanced_assignment(token_expert_affinities)
            if is_training
            else self.greedy_assignment(token_expert_affinities)
        )
        # Swap these tokens for the right ones for our expert
        routed_features = All2All.apply(
            features[sort_by_expert], input_splits, output_splits
        )

        if routed_features.size(0) > 0:
            # Mix in the expert network based on how appropriate it is for these tokens
            alpha = torch.sigmoid(
                routed_features.mv(self.expert_centroids[self.expert_id])
            ).unsqueeze(1)
            routed_features = (
                alpha * self.expert_network(routed_features)
                + (1 - alpha) * routed_features
            )
        # Return to original worker and ordering
        result = All2All.apply(routed_features, output_splits, input_splits)[
            self.inverse_sort(sort_by_expert)
        ]

        if self.shuffle and is_training:
            # Undo shuffling
            result = All2All.apply(
                result, shuffle_output_splits.tolist(), shuffle_input_splits.tolist()
            )[self.inverse_sort(shuffle_sort)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(input_features.size()), None, None, None

    def auction_lap(self, X, eps=None, compute_score=True):
        """
        X: n-by-n matrix w/ integer entries
        eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
        """
        eps = 1 / X.shape[0] if eps is None else eps

        # --
        # Init
        cost = torch.zeros((1, X.shape[1]))
        curr_ass = torch.zeros(X.shape[0]).long() - 1
        bids = torch.zeros(X.shape)

        if X.is_cuda:
            cost, curr_ass, bids = cost.cuda(), curr_ass.cuda(), bids.cuda()

        counter = 0
        while (curr_ass == -1).any():
            counter += 1

            # --
            # Bidding

            unassigned = (curr_ass == -1).nonzero().squeeze()
            if unassigned.dim() == 0:
                unassigned = torch.tensor([unassigned.item()])

            if counter > 100:
                curr_ass[curr_ass == -1] = 0
                break

            value = X[unassigned] - cost
            top_value, top_idx = value.topk(2, dim=1)

            first_idx = top_idx[:, 0]
            first_value, second_value = top_value[:, 0], top_value[:, 1]

            bid_increments = first_value - second_value + eps

            bids_ = bids[unassigned]
            bids_.zero_()
            try:
                bids_.scatter_(
                    dim=1,
                    index=first_idx.contiguous().view(-1, 1),
                    src=bid_increments.view(-1, 1),
                )
            except:
                torch.set_printoptions(profile="full")

            # --
            # Assignment

            have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()

            high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)
            high_bidders = unassigned[high_bidders.squeeze()]

            cost[:, have_bidder] += high_bids

            curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
            curr_ass[high_bidders] = have_bidder.squeeze()

        score = None
        if compute_score:
            score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())

        return score, curr_ass, counter

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        results = torch.empty_like(order).scatter_(
            0, order, torch.arange(0, order.size(0), device=order.device)
        )
        return results

    def balanced_assignment(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        unsorted = self.auction_lap(scores)[1]
        nested = [[] for _ in range(self.num_workers)]
        for i, x in enumerate(unsorted.tolist()):
            nested[x].append(i)

        unnested = [n for inner in nested for n in inner]
        input_splits = [len(inner) for inner in nested]
        output_splits = All2All.apply(torch.tensor(input_splits, device=scores.device))

        return torch.tensor(unnested), input_splits, output_splits.tolist()

    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros(
            (self.num_workers,), dtype=torch.long, device=scores.device
        )
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2All.apply(output_splits)
        return worker2token, output_splits.tolist(), input_splits.tolist()


class BaseSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu") or "relu"
        )
        self.norm = LayerNorm(args.decoder_embed_dim, export=False)
        self.ff1 = torch.nn.Linear(args.decoder_embed_dim, args.decoder_ffn_embed_dim)
        self.ff2 = torch.nn.Linear(args.decoder_ffn_embed_dim, args.decoder_embed_dim)
        self.ff2.weight.data.zero_()

    def forward(self, xs):
        return xs + self.ff2(self.activation_fn(self.ff1(self.norm(xs))))


# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = (
            torch.empty_like(xs)
            if output_splits is None
            else xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        )
        torch.distributed.all_to_all_single(
            ys, xs, output_split_sizes=output_splits, input_split_sizes=input_splits
        )
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = (
            torch.empty_like(grad_output)
            if ctx.input_splits is None
            else grad_output.new_empty(
                size=[sum(ctx.input_splits)] + list(grad_output.size()[1:])
            )
        )
        torch.distributed.all_to_all_single(
            result,
            grad_output,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
        )
        return result, None, None
