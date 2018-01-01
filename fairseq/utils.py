# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import contextlib
import logging
import os
import torch
import traceback
import sys

from torch.autograd import Variable
from torch.serialization import default_restore_location

from fairseq import criterions, data, models, progress_bar, tokenizer


def parse_args_and_arch(parser):
    args = parser.parse_args()
    args.model = models.arch_model_map[args.arch]
    args = getattr(models, args.model).parse_arch(args)
    return args


def build_model(args, src_dict, dst_dict):
    assert hasattr(models, args.model), 'Missing model type'
    return getattr(models, args.model).build_model(args, src_dict, dst_dict)


def build_criterion(args, src_dict, dst_dict):
    if args.label_smoothing > 0:
        return criterions.LabelSmoothedCrossEntropyCriterion(args, dst_dict)
    else:
        return criterions.CrossEntropyCriterion(args, dst_dict)


def build_progress_bar(args, iterator, epoch=None, prefix=None):
    if args.log_format is None:
        args.log_format = 'tqdm' if sys.stderr.isatty() else 'simple'

    if args.log_format == 'json':
        bar = progress_bar.json_progress_bar(iterator, epoch, prefix, args.log_interval)
    elif args.log_format == 'none':
        bar = progress_bar.noop_progress_bar(iterator, epoch, prefix)
    elif args.log_format == 'simple':
        bar = progress_bar.simple_progress_bar(iterator, epoch, prefix, args.log_interval)
    elif args.log_format == 'tqdm':
        bar = progress_bar.tqdm_progress_bar(iterator, epoch, prefix)
    else:
        raise ValueError('Unknown log format: {}'.format(args.log_format))
    return bar


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def save_state(filename, args, model, criterion, optimizer, lr_scheduler, optim_history=None, extra_state=None):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    state_dict = {
        'args': args,
        'model': model.state_dict(),
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'best_loss': lr_scheduler.best,
            }
        ],
        'last_optimizer_state': optimizer.state_dict(),
        'extra_state': extra_state,
    }
    torch_persistent_save(state_dict, filename)


def load_model_state(filename, model, cuda_device=None):
    if not os.path.exists(filename):
        return None, [], None
    if cuda_device is None:
        state = torch.load(filename)
    else:
        state = torch.load(
            filename,
            map_location=lambda s, l: default_restore_location(s, 'cuda:{}'.format(cuda_device))
        )
    state = _upgrade_state_dict(state)
    state['model'] = model.upgrade_state_dict(state['model'])

    # load model parameters
    try:
        model.load_state_dict(state['model'])
    except:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    return state['extra_state'], state['optimizer_history'], state['last_optimizer_state']


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    # add optimizer_history
    if 'optimizer_history' not in state:
        state['optimizer_history'] = [
            {
                'criterion_name': criterions.CrossEntropyCriterion.__name__,
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
    return state


def load_ensemble_for_inference(filenames, src_dict=None, dst_dict=None, data_dir=None):
    """Load an ensemble of models for inference.

    The source and target dictionaries can be given explicitly, or loaded from
    the `data_dir` directory.
    """
    # load model architectures and weights
    states = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        states.append(
            torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        )
    args = states[0]['args']
    args = _upgrade_args(args)

    if src_dict is None or dst_dict is None:
        assert data_dir is not None
        src_dict, dst_dict = data.load_dictionaries(data_dir, args.source_lang, args.target_lang)

    # build ensemble
    ensemble = []
    for state in states:
        model = build_model(args, src_dict, dst_dict)
        state['model'] = model.upgrade_state_dict(state['model'])
        model.load_state_dict(state['model'])
        ensemble.append(model)
    return ensemble, args


def _upgrade_args(args):
    if not hasattr(args, 'max_source_positions'):
        args.max_source_positions = args.max_positions
        args.max_target_positions = args.max_positions
    return args


def make_variable(sample, volatile=False, cuda_device=None):
    """Wrap input tensors in Variable class."""

    def _make_variable(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            if cuda_device is not None and torch.cuda.is_available():
                maybe_tensor = maybe_tensor.cuda(async=True, device=cuda_device)
            return Variable(maybe_tensor, volatile=volatile)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _make_variable(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_make_variable(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _make_variable(sample)


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str):
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, 'r') as f:
            for line in f:
                l = line.split()
                align_dict[l[0]] = l[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    # Tokens are strings here
    hypo_tokens = tokenizer.tokenize_line(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = tokenizer.tokenize_line(src_str) + ['<eos>']
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return ' '.join(hypo_tokens)


def post_process_prediction(hypo_tokens, src_str, alignment, align_dict, dst_dict, remove_bpe):
    hypo_str = dst_dict.string(hypo_tokens, remove_bpe)
    if align_dict is not None:
        hypo_str = replace_unk(hypo_str, src_str, alignment, align_dict, dst_dict.unk_string())
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tokenizer.Tokenizer.tokenize(hypo_str, dst_dict, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def lstrip_pad(tensor, pad):
    return tensor[tensor.eq(pad).long().sum():]


def rstrip_pad(tensor, pad):
    strip = tensor.eq(pad).long().sum()
    if strip > 0:
        return tensor[:-strip]
    return tensor


def strip_pad(tensor, pad):
    if tensor[0] == pad:
        tensor = lstrip_pad(tensor, pad)
    if tensor[-1] == pad:
        tensor = rstrip_pad(tensor, pad)
    return tensor


def maybe_no_grad(condition):
    if hasattr(torch, 'no_grad') and condition:
        return torch.no_grad()
    # no-op context manager
    return contextlib.ExitStack()
