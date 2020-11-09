# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple, Union

import torch

from fairseq import utils


def checkpoint_wrapper(m):
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:
    - wraps an nn.Module, so that all subsequent calls will use checkpointing
    - handles keyword arguments in the forward
    - handles non-Tensor outputs from the forward

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))
    """
    original_forward = m.forward

    def _checkpointed_forward(*args, **kwargs):
        # Autograd Functions in PyTorch work best with positional args, since
        # the backward must return gradients (or None) for every input argument.
        # We can flatten keyword arguments to make this easier.
        kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
        parent_ctx_dict = {}
        output = CheckpointFunction.apply(
            original_forward, parent_ctx_dict, kwarg_keys, *flat_args
        )
        if isinstance(output, torch.Tensor):
            return output
        else:
            packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
            if packed_non_tensor_outputs:
                output = unpack_non_tensors(output, packed_non_tensor_outputs)
            return output

    m.forward = _checkpointed_forward
    return m


def pack_kwargs(*args, **kwargs) -> Tuple[List[str], List[Any]]:
    """
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == [1, 2]
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys = []
    flat_args = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return kwarg_keys, flat_args


def unpack_kwargs(
    kwarg_keys: List[str], flat_args: List[Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs


def split_non_tensors(
    mixed: Union[torch.Tensor, Tuple[Any]]
) -> Tuple[Tuple[torch.Tensor], Dict[str, List[Any]]]:
    """
    Usage::

        x = torch.Tensor([1])
        y = torch.Tensor([2])
        tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
        recon = unpack_non_tensors(tensors, packed_non_tensors)
        assert recon == (x, y, None, 3)
    """
    if isinstance(mixed, torch.Tensor):
        return (mixed,), None
    tensors = []
    packed_non_tensors = {"is_tensor": [], "objects": []}
    for o in mixed:
        if isinstance(o, torch.Tensor):
            packed_non_tensors["is_tensor"].append(True)
            tensors.append(o)
        else:
            packed_non_tensors["is_tensor"].append(False)
            packed_non_tensors["objects"].append(o)
    return tuple(tensors), packed_non_tensors


def unpack_non_tensors(
    tensors: Tuple[torch.Tensor],
    packed_non_tensors: Dict[str, List[Any]],
) -> Tuple[Any]:
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict)
    mixed = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list)
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)


class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.

    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling ``unpack_non_tensors``.
    """

    @staticmethod
    def forward(ctx, run_function, parent_ctx_dict, kwarg_keys, *args):
        if torch.is_grad_enabled():  # grad may be disabled, e.g., during validation
            torch.utils.checkpoint.check_backward_validity(args)

        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = utils.get_rng_state()

        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        ctx.save_for_backward(*tensor_inputs)
        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs

        with torch.no_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)

        if isinstance(outputs, torch.Tensor):
            return outputs
        else:
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict["packed_non_tensor_outputs"] = packed_non_tensor_outputs
            return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        tensor_inputs = ctx.saved_tensors
        tensor_inputs = torch.utils.checkpoint.detach_variable(tensor_inputs)
        inputs = unpack_non_tensors(tensor_inputs, ctx.packed_non_tensor_inputs)

        # Store the current states.
        bwd_rng_state = utils.get_rng_state()

        # Set the states to what it used to be before the forward pass.
        utils.set_rng_state(ctx.fwd_rng_state)

        with torch.enable_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(ctx.kwarg_keys, inputs)
            outputs = ctx.run_function(*unpacked_args, **unpacked_kwargs)
            tensor_outputs, _ = split_non_tensors(outputs)

        # Set the states back to what it was at the start of this function.
        utils.set_rng_state(bwd_rng_state)

        # Run backward() with only Tensors that require grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(tensor_outputs)):
            if tensor_outputs[i].requires_grad:
                outputs_with_grad.append(tensor_outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of the outputs have requires_grad=True, "
                "this checkpoint() is not necessary"
            )

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs
        )
        return (None, None, None) + grads
