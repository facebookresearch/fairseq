# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def checkpoint_wrapper(module, *args, **kwargs):
    try:
        from fairscale.nn.misc.checkpoint_activations import checkpoint_wrapper as _checkpoint_wrapper
    except ImportError:
        try:
            from fairscale.nn import checkpoint_wrapper as _checkpoint_wrapper
        except ImportError:
            raise ImportError(
                "Cannot find fairscale.nn.misc.checkpoint_activations. "
                "Please install fairscale with: pip install fairscale"
            )

    module = _checkpoint_wrapper(module, *args, **kwargs)

    if hasattr(module, "extra_repr"):
        orig_extra_repr = module.extra_repr
    else:
        orig_extra_repr = None

    def extra_repr():
        return f"[checkpointed] {orig_extra_repr()}" if orig_extra_repr is not None else ""

    module.extra_repr = extra_repr

    return module
