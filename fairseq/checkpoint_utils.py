# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import os
import re
import traceback
from collections import OrderedDict
from typing import Union

import torch
from fairseq.file_io import PathManager
from fairseq.models import FairseqDecoder, FairseqEncoder
from torch.serialization import default_restore_location


logger = logging.getLogger(__name__)


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    from fairseq import distributed_utils, meters

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if args.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if args.no_save or not distributed_utils.is_master(args):
        return

    def is_better(a, b):
        return a >= b if args.maximize_best_checkpoint_metric else a <= b

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}.pt".format(epoch)] = (
        end_of_epoch
        and not args.no_epoch_checkpoints
        and epoch % args.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}.pt".format(epoch, updates)] = (
        not end_of_epoch
        and args.save_interval_updates > 0
        and updates % args.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best.pt"] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )
    if val_loss is not None and args.keep_best_checkpoints > 0:
        checkpoint_conds["checkpoint.best_{}_{:.2f}.pt".format(
            args.best_checkpoint_metric, val_loss)] = (
            not hasattr(save_checkpoint, "best")
            or is_better(val_loss, save_checkpoint.best)
        )
    checkpoint_conds["checkpoint_last.pt"] = not args.no_last_checkpoints

    extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})

    checkpoints = [
        os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            PathManager.copy(checkpoints[0], cp, overwrite=True)

        write_timer.stop()
        logger.info(
            "saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r"checkpoint_\d+_(\d+)\.pt"
        )
        for old_chk in checkpoints[args.keep_interval_updates :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(args.save_dir, pattern=r"checkpoint(\d+)\.pt")
        for old_chk in checkpoints[args.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_best_checkpoints > 0:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r"checkpoint\.best_{}_(\d+\.?\d*)\.pt".format(args.best_checkpoint_metric))
        if not args.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[args.keep_best_checkpoints:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(args, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """
    # only one worker should attempt to create the required dir
    if args.distributed_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.restore_file == "checkpoint_last.pt":
        checkpoint_path = os.path.join(args.save_dir, "checkpoint_last.pt")
    else:
        checkpoint_path = args.restore_file

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        args.reset_optimizer,
        args.reset_lr_scheduler,
        eval(args.optimizer_overrides),
        reset_meters=args.reset_meters,
    )

    if (
        extra_state is not None
        and "best" in extra_state
        and not args.reset_optimizer
        and not args.reset_meters
    ):
        save_checkpoint.best = extra_state["best"]

    if extra_state is not None and not args.reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state["train_iterator"]
        epoch_itr = trainer.get_train_iterator(
            epoch=itr_state["epoch"], load_dataset=True, **passthrough_args
        )
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(
            epoch=1, load_dataset=True, **passthrough_args
        )

    trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch_itr


def load_checkpoint_to_cpu(path, arg_overrides=None):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility)."""
    with PathManager.open(path, "rb") as f:
        state = torch.load(
            f, map_location=lambda s, l: default_restore_location(s, "cpu")
        )

    args = state["args"]
    if arg_overrides is not None:
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
    state = _upgrade_state_dict(state)
    return state


def load_model_ensemble(filenames, arg_overrides=None, task=None):
    """Loads an ensemble of models.

    Args:
        filenames (List[str]): checkpoint files to load
        arg_overrides (Dict[str,Any], optional): override model args that
            were used during model training
        task (fairseq.tasks.FairseqTask, optional): task to use for loading
    """
    ensemble, args, _task = load_model_ensemble_and_task(filenames, arg_overrides, task)
    return ensemble, args


def load_model_ensemble_and_task(filenames, arg_overrides=None, task=None):
    from fairseq import tasks

    ensemble = []
    for filename in filenames:
        if not PathManager.exists(filename):
            raise IOError("Model file not found: {}".format(filename))
        state = load_checkpoint_to_cpu(filename, arg_overrides)

        args = state["args"]
        if task is None:
            task = tasks.setup_task(args)

        # build model for ensemble
        model = task.build_model(args)
        model.load_state_dict(state["model"], strict=True, args=args)
        ensemble.append(model)
    return ensemble, args, task


def checkpoint_paths(path, pattern=r"checkpoint(\d+)\.pt"):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logger.error(traceback.format_exc())


def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict


def save_state(
    filename,
    args,
    model_state_dict,
    criterion,
    optimizer,
    lr_scheduler,
    num_updates,
    optim_history=None,
    extra_state=None,
):
    from fairseq import utils

    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    state_dict = {
        "args": args,
        "model": model_state_dict if model_state_dict else {},
        "optimizer_history": optim_history
        + [
            {
                "criterion_name": criterion.__class__.__name__,
                "optimizer_name": optimizer.__class__.__name__,
                "lr_scheduler_state": lr_scheduler.state_dict(),
                "num_updates": num_updates,
            }
        ],
        "extra_state": extra_state,
    }
    if utils.has_parameters(criterion):
        state_dict["criterion"] = criterion.state_dict()
    if not args.no_save_optimizer_state:
        state_dict["last_optimizer_state"] = convert_state_dict_type(
            optimizer.state_dict()
        )

    with PathManager.open(filename, "wb") as f:
        torch_persistent_save(state_dict, f)


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    from fairseq import models, registry, tasks

    # add optimizer_history
    if "optimizer_history" not in state:
        state["optimizer_history"] = [
            {"criterion_name": "CrossEntropyCriterion", "best_loss": state["best_loss"]}
        ]
        state["last_optimizer_state"] = state["optimizer"]
        del state["optimizer"]
        del state["best_loss"]
    # move extra_state into sub-dictionary
    if "epoch" in state and "extra_state" not in state:
        state["extra_state"] = {
            "epoch": state["epoch"],
            "batch_offset": state["batch_offset"],
            "val_loss": state["val_loss"],
        }
        del state["epoch"]
        del state["batch_offset"]
        del state["val_loss"]
    # reduce optimizer history's memory usage (only keep the last state)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1]["optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    # record the optimizer class name
    if "optimizer_name" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["optimizer_name"] = "FairseqNAG"
    # move best_loss into lr_scheduler_state
    if "lr_scheduler_state" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["lr_scheduler_state"] = {
            "best": state["optimizer_history"][-1]["best_loss"]
        }
        del state["optimizer_history"][-1]["best_loss"]
    # keep track of number of updates
    if "num_updates" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["num_updates"] = 0
    # old model checkpoints may not have separate source/target positions
    if hasattr(state["args"], "max_positions") and not hasattr(
        state["args"], "max_source_positions"
    ):
        state["args"].max_source_positions = state["args"].max_positions
        state["args"].max_target_positions = state["args"].max_positions
    # use stateful training data iterator
    if "train_iterator" not in state["extra_state"]:
        state["extra_state"]["train_iterator"] = {
            "epoch": state["extra_state"]["epoch"],
            "iterations_in_epoch": state["extra_state"].get("batch_offset", 0),
        }
    # default to translation task
    if not hasattr(state["args"], "task"):
        state["args"].task = "translation"
    # --raw-text and --lazy-load are deprecated
    if getattr(state["args"], "raw_text", False):
        state["args"].dataset_impl = "raw"
    elif getattr(state["args"], "lazy_load", False):
        state["args"].dataset_impl = "lazy"
    # epochs start at 1
    state["extra_state"]["train_iterator"]["epoch"] = max(
        state["extra_state"]["train_iterator"].get("epoch", 1),
        1,
    )

    # set any missing default values in the task, model or other registries
    registry.set_defaults(state["args"], tasks.TASK_REGISTRY[state["args"].task])
    registry.set_defaults(state["args"], models.ARCH_MODEL_REGISTRY[state["args"].arch])
    for registry_name, REGISTRY in registry.REGISTRIES.items():
        choice = getattr(state["args"], registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            registry.set_defaults(state["args"], cls)

    return state


def prune_state_dict(state_dict, args):
    """Prune the given state_dict if desired for LayerDrop
    (https://arxiv.org/abs/1909.11556).

    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.

    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    if not args or args.arch == "ptt_transformer":
        # args should not be none, but don't crash if it is.
        return state_dict

    encoder_layers_to_keep = (
        args.encoder_layers_to_keep if "encoder_layers_to_keep" in vars(args) else None
    )
    decoder_layers_to_keep = (
        args.decoder_layers_to_keep if "decoder_layers_to_keep" in vars(args) else None
    )

    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict

    # apply pruning
    logger.info(
        "Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop"
    )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(
            [int(layer_string) for layer_string in layers_to_keep.split(",")]
        )
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)

        regex = re.compile("^{layer}.*\.layers\.(\d+)".format(layer=layer_name))
        return {"substitution_regex": regex, "mapping_dict": mapping_dict}

    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search("\.layers\.(\d+)\.", layer_name)
        # if layer has no number in it, it is a supporting layer, such as an
        # embedding
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        # otherwise, layer should be pruned.
        original_layer_number = match.group(1)
        # figure out which mapping dict to replace from
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass[
                "substitution_regex"
            ].search(layer_name):
                new_layer_number = pruning_pass["mapping_dict"][original_layer_number]
                substitution_match = pruning_pass["substitution_regex"].search(
                    layer_name
                )
                new_state_key = (
                    layer_name[: substitution_match.start(1)]
                    + new_layer_number
                    + layer_name[substitution_match.end(1) :]
                )
                new_state_dict[new_state_key] = state_dict[layer_name]

    # Since layers are now pruned, *_layers_to_keep are no longer needed.
    # This is more of "It would make it work fix" rather than a proper fix.
    if "encoder_layers_to_keep" in vars(args):
        args.encoder_layers_to_keep = None
    if "decoder_layers_to_keep" in vars(args):
        args.decoder_layers_to_keep = None

    return new_state_dict


def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder], checkpoint: str
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            component_state_dict[component_subkey] = state["model"][key]
    component.load_state_dict(component_state_dict, strict=True)
    return component


def verify_checkpoint_directory(save_dir: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.join(save_dir, "dummy")
    try:
        with open(temp_file_path, "w"):
            pass
    except OSError as e:
        logger.warning("Unable to access checkpoint save directory: {}".format(save_dir))
        raise e
    else:
        os.remove(temp_file_path)
