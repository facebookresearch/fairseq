# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse
from collections import OrderedDict
from typing import Union
import collections
import logging
import os
import re
import traceback
import shutil

import torch
from torch.serialization import default_restore_location

from fairseq.models import FairseqEncoder, FairseqDecoder


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    from fairseq import distributed_utils, meters

    if args.no_save or not distributed_utils.is_master(args):
        return

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
        end_of_epoch and not args.no_epoch_checkpoints and
        epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
        not end_of_epoch and args.save_interval_updates > 0 and
        updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
        val_loss is not None and
        (not hasattr(save_checkpoint, 'best') or is_better(val_loss, save_checkpoint.best))
    )
    checkpoint_conds['checkpoint_last.pt'] = not args.no_last_checkpoints

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = is_better(val_loss, prev_best)
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            shutil.copyfile(checkpoints[0], cp)

        write_timer.stop()
        print('| saved checkpoint {} (epoch {} @ {} updates) (writing took {} seconds)'.format(
            checkpoints[0], epoch, updates, write_timer.sum))

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt',
        )
        for old_chk in checkpoints[args.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r'checkpoint(\d+)\.pt',
        )
        for old_chk in checkpoints[args.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(args, trainer):
    """Load a checkpoint and restore the training iterator."""
    # only one worker should attempt to create the required dir
    if args.distributed_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    if os.path.isabs(args.restore_file):
        checkpoint_path = args.restore_file
    else:
        checkpoint_path = os.path.join(args.save_dir, args.restore_file)

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        args.reset_optimizer,
        args.reset_lr_scheduler,
        eval(args.optimizer_overrides),
        reset_meters=args.reset_meters,
    )

    if (
        extra_state is not None
        and 'best' in extra_state
        and not args.reset_optimizer
        and not args.reset_meters
    ):
        save_checkpoint.best = extra_state['best']

    if extra_state is not None and not args.reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state['train_iterator']
        epoch_itr = trainer.get_train_iterator(epoch=itr_state['epoch'])
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(epoch=0)

    trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch_itr


def load_checkpoint_to_cpu(path, arg_overrides=None):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility)."""
    state = torch.load(
        path, map_location=lambda s, l: default_restore_location(s, 'cpu'),
    )
    args = state['args']
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
    ensemble, args, _task = _load_model_ensemble(filenames, arg_overrides, task)
    return ensemble, args


def _load_model_ensemble(filenames, arg_overrides=None, task=None):
    from fairseq import tasks

    ensemble = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        state = load_checkpoint_to_cpu(filename, arg_overrides)

        args = state['args']
        if task is None:
            task = tasks.setup_task(args)

        # build model for ensemble
        model = task.build_model(args)
        model.load_state_dict(state['model'], strict=True)
        ensemble.append(model)
    return ensemble, args, task


def checkpoint_paths(path, pattern=r'checkpoint(\d+)\.pt'):
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
            idx = int(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


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
    filename, args, model_state_dict, criterion, optimizer, lr_scheduler,
    num_updates, optim_history=None, extra_state=None,
):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    state_dict = {
        'args': args,
        'model': model_state_dict if model_state_dict else {},
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer_name': optimizer.__class__.__name__,
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'num_updates': num_updates,
            }
        ],
        'extra_state': extra_state,
    }
    if not args.no_save_optimizer_state:
        state_dict['last_optimizer_state'] = convert_state_dict_type(optimizer.state_dict())
    torch_persistent_save(state_dict, filename)


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    from fairseq import models, registry, tasks

    # add optimizer_history
    if 'optimizer_history' not in state:
        state['optimizer_history'] = [
            {
                'criterion_name': 'CrossEntropyCriterion',
                'best_loss': state['best_loss'],
            },
        ]
        state['last_optimizer_state'] = state['optimizer']
        del state['optimizer']
        del state['best_loss']
    # move extra_state into sub-dictionary
    if 'epoch' in state and 'extra_state' not in state:
        state['extra_state'] = {
            'epoch': state['epoch'],
            'batch_offset': state['batch_offset'],
            'val_loss': state['val_loss'],
        }
        del state['epoch']
        del state['batch_offset']
        del state['val_loss']
    # reduce optimizer history's memory usage (only keep the last state)
    if 'optimizer' in state['optimizer_history'][-1]:
        state['last_optimizer_state'] = state['optimizer_history'][-1]['optimizer']
        for optim_hist in state['optimizer_history']:
            del optim_hist['optimizer']
    # record the optimizer class name
    if 'optimizer_name' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['optimizer_name'] = 'FairseqNAG'
    # move best_loss into lr_scheduler_state
    if 'lr_scheduler_state' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['lr_scheduler_state'] = {
            'best': state['optimizer_history'][-1]['best_loss'],
        }
        del state['optimizer_history'][-1]['best_loss']
    # keep track of number of updates
    if 'num_updates' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['num_updates'] = 0
    # old model checkpoints may not have separate source/target positions
    if hasattr(state['args'], 'max_positions') and not hasattr(state['args'], 'max_source_positions'):
        state['args'].max_source_positions = state['args'].max_positions
        state['args'].max_target_positions = state['args'].max_positions
    # use stateful training data iterator
    if 'train_iterator' not in state['extra_state']:
        state['extra_state']['train_iterator'] = {
            'epoch': state['extra_state']['epoch'],
            'iterations_in_epoch': state['extra_state'].get('batch_offset', 0),
        }
    # default to translation task
    if not hasattr(state['args'], 'task'):
        state['args'].task = 'translation'

    def set_defaults(cls):
        if not hasattr(cls, 'add_args'):
            return
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, allow_abbrev=False)
        cls.add_args(parser)
        # copied from argparse.py:
        defaults = argparse.Namespace()
        for action in parser._actions:
            if action.dest is not argparse.SUPPRESS:
                if not hasattr(defaults, action.dest):
                    if action.default is not argparse.SUPPRESS:
                        setattr(defaults, action.dest, action.default)
        for key, default_value in vars(defaults).items():
            if not hasattr(state['args'], key):
                setattr(state['args'], key, default_value)

    # set any missing default values in the task, model or other registries
    set_defaults(tasks.TASK_REGISTRY[state['args'].task])
    set_defaults(models.ARCH_MODEL_REGISTRY[state['args'].arch])
    for registry_name, REGISTRY in registry.REGISTRIES.items():
        choice = getattr(state['args'], registry_name, None)
        if choice is not None:
            cls = REGISTRY['registry'][choice]
            set_defaults(cls)

    return state


def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder], checkpoint: str
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not os.path.exists(checkpoint):
        raise IOError('Model file not found: {}'.format(checkpoint))
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
            component_subkey = key[len(component_type) + 1:]
            component_state_dict[component_subkey] = state["model"][key]
    component.load_state_dict(component_state_dict, strict=True)
    return component
