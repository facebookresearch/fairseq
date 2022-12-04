"""Copied from torch.hub, but modified to support reading from the local checkout."""

import os
import sys

import torch.hub


MODULE_HUBCONF = "hubconf.py"


# Ideally this should be `def load(github, model, *args, forece_reload=False, **kwargs):`,
# but Python2 complains syntax error for it. We have to skip force_reload in function
# signature here but detect it in kwargs instead.
# TODO: fix it after Python2 EOL
def load(model, *args, **kwargs):
    r"""
    Load a model from a github repo, with pretrained weights.

    Args:
        model: Required, a string of entrypoint name defined in repo's hubconf.py
        *args: Optional, the corresponding args for callable `model`.
        force_reload: Optional, whether to force a fresh download of github repo unconditionally.
            Default is `False`.
        **kwargs: Optional, the corresponding kwargs for callable `model`.

    Returns:
        a single model with corresponding pretrained weights.

    Example:
        >>> model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    """
    # Setup hub_dir to save downloaded files
    _setup_hubdir()

    force_reload = kwargs.get("force_reload", False)
    kwargs.pop("force_reload", None)

    repo_dir = os.path.dirname(os.path.dirname(__file__))

    sys.path.insert(0, repo_dir)

    hub_module = torch.hub.import_module(
        MODULE_HUBCONF, repo_dir + "/" + MODULE_HUBCONF
    )

    entry = torch.hub._load_entry_from_hubconf(hub_module, model)

    model = entry(*args, **kwargs)

    sys.path.remove(repo_dir)

    return model


def _setup_hubdir():
    if hasattr(torch.hub, "_setup_hubdir"):
        torch.hub._setup_hubdir()
    else:
        hub_dir = torch.hub.get_dir()
        if not os.path.exists(hub_dir):
            os.makedirs(hub_dir)


def list(force_reload=False):
    r"""
    List all entrypoints available in `github` hubconf.

    Args:
        force_reload: Optional, whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Returns:
        entrypoints: a list of available entrypoint names

    Example:
        >>> entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    """
    # Setup hub_dir to save downloaded files
    _setup_hubdir()

    repo_dir = os.path.dirname(os.path.dirname(__file__))

    sys.path.insert(0, repo_dir)

    hub_module = torch.hub.import_module(
        MODULE_HUBCONF, repo_dir + "/" + MODULE_HUBCONF
    )

    sys.path.remove(repo_dir)

    # We take functions starts with '_' as internal helper functions
    entrypoints = [
        f
        for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith("_")
    ]

    return entrypoints
