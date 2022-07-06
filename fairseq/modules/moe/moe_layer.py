# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.cuda import Event as CudaEvent
from torch.nn import Module, ModuleList

from fairseq import distributed_utils
from fairseq.modules.linear import Linear

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

logger = logging.getLogger(__name__)

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe

    has_tutel = True
except (ModuleNotFoundError, AssertionError):
    # import raises AssertionError without CUDA
    has_tutel = False


def get_fused_cumsum_sub_one(use_tutel):
    if use_tutel:
        return tutel_moe.fast_cumsum_sub_one
    return lambda mask: torch.cumsum(mask, dim=0) - 1


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(
        self,
        gate: Module,
        experts: Union[Module, ModuleList],
        args,
        group: Optional[Any] = None,
        all2all_group: Optional[Any] = None,
        max_positions: Optional[int] = None,
        init_model_on_gpu: Optional[bool] = False,
        tok_dropout: float = 0.0,
        moe_local_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.expert_group = (
            group
            if group is not None
            else distributed_utils.get_moe_group(args.moe_expert_count)
        )
        self.all2all_group = (
            all2all_group
            if all2all_group is not None
            else distributed_utils.get_all2all_group(args.moe_expert_count)
        )
        for p in experts.parameters():
            p.expert = True  # type: ignore
        self.world_size = distributed_utils.get_world_size(self.expert_group)
        self.all2all_size = distributed_utils.get_world_size(self.all2all_group)
        self.num_local_experts = len(self.experts)
        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0
        self.use_tutel = getattr(args, "use_tutel_moe", False)
        self.moe_eval_capacity_max_seqlen = getattr(
            args, "moe_eval_capacity_max_seqlen", False
        )
        self.max_positions = max_positions
        self.tok_dropout = tok_dropout
        self.moe_local_drop = moe_local_drop

    def forward(
        self, *input: Tensor, input_padding_mask=None, prefix_tokens=None, **kwargs: Any
    ) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert (
            len(input.shape) == 3
        ), "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        if input_padding_mask is not None:
            assert (
                len(input_padding_mask.shape) == 2
            ), "input Tensor must have dimensions: (s)equence, (t)oken"
            assert input_padding_mask.shape[0] == input.shape[0]
            assert input_padding_mask.shape[1] == input.shape[1]
        # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input.shape[2]
        # Pad to expected batch size
        input_shape = list(input.shape)
        expected_bsz = (
            getattr(self.args, "batch_size", 0)
            if self.training
            else getattr(self.args, "batch_size_valid", 0)
        )
        # This indicates that --batch-size or --max-sentences is not specified
        if expected_bsz is None:
            expected_bsz = 0
        # Note: Padding is not necessary at generation time at present
        # because all DDP workers process the same batch. Also, batch size at generation time
        # can be different from that present in the checkpoint state
        if (
            not self.in_generation
            and expected_bsz != 0
            and input_shape[0] != expected_bsz
        ):
            logger.warning(
                f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})"
            )
            assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
            padded_input = torch.zeros(
                (expected_bsz, input_shape[1], input_shape[2]),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            padded_input[: input_shape[0], :, :] = input
            input = padded_input

            padded_input_padding_mask = torch.ones(
                (
                    expected_bsz,
                    input_shape[1],
                ),
                dtype=torch.bool,
                device=input.device,
            )
            if input_padding_mask is not None:
                padded_input_padding_mask[: input_shape[0], :] = input_padding_mask
            else:
                padded_input_padding_mask[: input_shape[0], :] = False
            input_padding_mask = padded_input_padding_mask
            if prefix_tokens is not None:
                padded_prefix_tokens = torch.zeros(
                    (expected_bsz, prefix_tokens.shape[1]),
                    dtype=input.dtype,
                    layout=input.layout,
                    device=input.device,
                )
                padded_prefix_tokens[: input_shape[0], :] = prefix_tokens
                prefix_tokens = padded_prefix_tokens

        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape
        reshaped_input_padding_mask = (
            input_padding_mask.reshape(-1) if input_padding_mask is not None else None
        )
        # used for tracking some properties per token
        reshaped_prefix_tokens = (
            prefix_tokens.repeat(1, input.shape[1]).reshape(-1)
            if prefix_tokens is not None
            else None
        )

        # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
        # Pro of --max-tokens: more flexible for MT variable sequence lengths
        # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
        if expected_bsz == 0:
            expected_dim = int(
                distributed_utils.all_reduce(
                    reshaped_input_shape[0]
                    * torch.ones((1,), dtype=torch.long, device=input.device),
                    group=dist.group.WORLD,
                    op="max",
                ).item()
            )
            padded_input = torch.zeros(
                (expected_dim, reshaped_input_shape[1]),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            padded_input[: reshaped_input_shape[0], :] = reshaped_input
            reshaped_input = padded_input

            padded_input_padding_mask = torch.ones(
                (expected_dim,), dtype=torch.bool, device=padded_input.device
            )
            if reshaped_input_padding_mask is not None:
                padded_input_padding_mask[
                    : reshaped_input_shape[0]
                ] = reshaped_input_padding_mask
            else:
                padded_input_padding_mask[: reshaped_input_shape[0]] = False
            reshaped_input_padding_mask = padded_input_padding_mask
            if reshaped_prefix_tokens is not None:
                padded_prefix_tokens = torch.zeros(
                    (expected_dim,),
                    dtype=prefix_tokens.dtype,
                    layout=prefix_tokens.layout,
                    device=prefix_tokens.device,
                )
                padded_prefix_tokens[: reshaped_input_shape[0]] = reshaped_prefix_tokens
                reshaped_prefix_tokens = padded_prefix_tokens

        if self.moe_eval_capacity_max_seqlen:
            eval_capacity_length = self.max_positions * input.shape[0]
        else:
            eval_capacity_length = None
        if self.use_tutel:
            l_aux, self.metadata, C, E, indices_, locations_, gates_ = self.gate(
                reshaped_input,
                reshaped_input_padding_mask,
                eval_capacity_length,
                prefix_tokens=reshaped_prefix_tokens,
            )
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, "_tutel_dispatcher"):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(
                    E, C, M, dispatch_dtype=reshaped_input.dtype
                )
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(
                reshaped_input,
                reshaped_input_padding_mask,
                eval_capacity_length,
                prefix_tokens=reshaped_prefix_tokens,
            )
            dispatch_mask = dispatch_mask.to(input.dtype).permute(
                1, 2, 0
            )  # S,E,C -> E,C,S
            E, C, S = dispatch_mask.size()
            M = reshaped_input.size(1)
            assert reshaped_input.size() == (S, M)
            # einsum("sec,sm->ecm")
            dispatched_input = torch.mm(
                dispatch_mask.view(E * C, S), reshaped_input
            )  # -> (E*C),M
        use_all_to_all = True
        if self.moe_local_drop > 0.0 and self.training:
            if dist.get_rank() == 0:
                use_all_to_all = (
                    dispatched_input.new_empty([]).uniform_() > self.moe_local_drop
                )
            else:
                use_all_to_all = dispatched_input.new_zeros([], dtype=torch.bool)
            distributed_utils.broadcast(use_all_to_all, src=0, group=self.all2all_group)
        if self.all2all_size > 1 and use_all_to_all:
            dispatched_input = self.all_to_all_wrapper(dispatched_input)
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(
            self.all2all_size, self.num_local_experts, -1, d_model
        )
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        if self.all2all_size > 1 and use_all_to_all:
            expert_output = self.all_to_all_wrapper(expert_output)
        if self.tok_dropout > 0.0:
            # TODO: replace w Dropout2d
            if self.training:
                # drop out 0.2 of token rembeddings
                mask = (
                    torch.empty(
                        expert_output.shape[:-1], device=expert_output.device
                    ).uniform_()
                    > self.tok_dropout
                )
                expert_output = mask.unsqueeze(-1) * expert_output
            else:
                expert_output = expert_output * (1 - self.tok_dropout)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            self.all2all_size * self.num_local_experts, -1, d_model
        )

        # einsum("sec,ecm->sm")
        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(
                expert_output.view(E * C, M)
            )
        else:
            # einsum("sec,ecm->sm")
            combined_output = combine_weights.view(S, E * C).mm(
                expert_output.view(E * C, M)
            )

        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[: reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[: input_shape[0], :, :]

        self.record_all_to_all_stats()

        return combined_output, {"moe_gate_loss": l_aux}

    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: Tensor):
        dummy_a2a = getattr(self.args, "dummy_a2a", False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        output = _AllToAll.apply(self.all2all_group, input)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += cpu_end - cpu_start
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def record_all_to_all_stats(self):
        # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
        record_a2a_perf_stats = getattr(self.args, "record_a2a_perf_stats", False)
        if record_a2a_perf_stats:
            torch.cuda.synchronize()
            self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
            a2a_cuda_time_ms = 0.0
            for ev_start, ev_end in self.a2a_cuda_event_intervals:
                a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
            self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
        # reset stats
        self.a2a_cpu_time_ms = 0.0
        self.a2a_cuda_event_intervals = []
