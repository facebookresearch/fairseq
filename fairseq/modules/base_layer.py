# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.distributed import utils as distributed_utils
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.linear import Linear


class BaseLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        ddp_backend = getattr(args, "ddp_backend")
        assert ddp_backend in {
            "no_c10d",
            "legacy_ddp",
        }, f"Incompatible backend {ddp_backend} for MOE models"

        self.num_workers = distributed_utils.get_data_parallel_world_size()
        self.expert_count = args.world_size
        self.num_local_experts = self.expert_count // self.num_workers

        expert_centroids = torch.empty(self.expert_count, args.decoder_embed_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter(
            "expert_centroids", torch.nn.Parameter(expert_centroids)
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    *([BaseSublayer(args) for _ in range(args.base_sublayers)])
                )
                for _ in range(self.num_local_experts)
            ]
        )

        self.expert_id = distributed_utils.get_data_parallel_rank()
        self.shuffle = args.base_shuffle
        self.cpp = self.load_assignment()

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.experts.parameters():
            param.base_expert = True

    def forward(self, input_features, *args, **kwargs):
        assert (
            not self.training or self.num_local_experts == 1
        ), f"Found {self.num_local_experts} local experts during training. \
             Can only have 1 during training."

        features = input_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad

        if self.shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = All2All.apply(features[shuffle_sort])

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

        # Merge splits to current world size so All2All works
        input_splits_merged = (
            self.merge_splits(self.expert_count, self.num_workers, input_splits)
            if not is_training
            else None
        )

        output_splits_merged = (
            self.merge_splits(self.expert_count, self.num_workers, output_splits)
            if not is_training
            else None
        )

        # Swap these tokens for the right ones for our expert
        routed_features = All2All.apply(
            features[sort_by_expert], output_splits_merged, input_splits_merged
        )

        if self.num_local_experts == 1:
            if routed_features.size(0) > 0:
                # Mix in the expert network based on how appropriate it is for these tokens
                alpha = torch.sigmoid(
                    routed_features.mv(self.expert_centroids[self.expert_id])
                ).unsqueeze(1)
                routed_features = (
                    alpha * self.experts[0](routed_features)
                    + (1 - alpha) * routed_features
                )
        else:
            start_index = 0
            for index, num_features_to_add in enumerate(input_splits):
                # Determine which local expert the features correspond to and then extract features
                local_expert_index = index % self.num_local_experts
                local_expert_features = routed_features[
                    start_index : start_index + num_features_to_add
                ]

                if local_expert_features.size(0) > 0:
                    alpha = torch.sigmoid(
                        local_expert_features.mv(
                            self.expert_centroids[
                                self.expert_id * self.num_local_experts
                                + local_expert_index
                            ]
                        )
                    ).unsqueeze(1)

                    local_expert_features = (
                        alpha * self.experts[local_expert_index](local_expert_features)
                        + (1 - alpha) * local_expert_features
                    )

                    routed_features[
                        start_index : start_index + num_features_to_add
                    ] = local_expert_features

                start_index += num_features_to_add

        # Return to original worker and ordering
        result = All2All.apply(
            routed_features, input_splits_merged, output_splits_merged
        )[self.inverse_sort(sort_by_expert)]

        if self.shuffle and is_training:
            # Undo shuffling
            result = All2All.apply(result)[self.inverse_sort(shuffle_sort)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(input_features.size()), None, None, None

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(
            0, order, torch.arange(0, order.size(0), device=order.device)
        )

    def balanced_assignment(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return self.cpp.balanced_assignment(scores), None, None

    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros(
            (self.expert_count,), dtype=torch.long, device=scores.device
        )
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2All.apply(output_splits)
        # Warning: .tolist() results in a device to host transfer from GPU -> CPU which is time-consuming
        # and slows down model training with FSDP ddp_backend
        return worker2token, input_splits.tolist(), output_splits.tolist()

    def merge_splits(self, world_size_train, world_size_inference, splits):
        assert world_size_train % world_size_inference == 0
        splits_stitched = list(splits)[0:world_size_inference]
        local_expert_count = int(world_size_train / world_size_inference)

        for i in range(0, world_size_train, local_expert_count):
            splits_stitched[i // local_expert_count] = sum(
                splits[i : i + local_expert_count]
            )

        return splits_stitched

    def load_assignment(self):
        try:
            from fairseq import libbase

            return libbase

        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing libbase. run `python setup.py build_ext --inplace`\n"
            )
            raise e


class BaseSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        init_model_on_gpu = getattr(args, "init_model_on_gpu", False)
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu") or "relu"
        )
        self.norm = LayerNorm(args.decoder_embed_dim, export=False)
        if init_model_on_gpu:
            self.norm = self.norm.cuda().half()
        self.ff1 = Linear(
            args.decoder_embed_dim,
            args.decoder_ffn_embed_dim,
            init_model_on_gpu=init_model_on_gpu,
        )
        self.ff2 = Linear(
            args.decoder_ffn_embed_dim,
            args.decoder_embed_dim,
            init_model_on_gpu=init_model_on_gpu,
        )
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
