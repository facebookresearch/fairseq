# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import defaultdict
import contextlib
import logging
import os
import torch
import traceback

from torch.autograd import Variable
from torch.serialization import default_restore_location

from fairseq import tokenizer


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def save_state(filename, args, model, criterion, optimizer, lr_scheduler,
               num_updates, optim_history=None, extra_state=None):
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
                'optimizer_name': optimizer.__class__.__name__,
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'num_updates': num_updates,
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
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    return state['extra_state'], state['optimizer_history'], state['last_optimizer_state']


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
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
    return state


def load_ensemble_for_inference(filenames, src_dict=None, dst_dict=None, data_dir=None):
    """Load an ensemble of models for inference.

    The source and target dictionaries can be given explicitly, or loaded from
    the `data_dir` directory.
    """
    from fairseq import data, models

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
        model = models.build_model(args, src_dict, dst_dict)
        model.load_state_dict(state['model'])
        ensemble.append(model)
    return ensemble, args


def _upgrade_args(args):
    if not hasattr(args, 'max_source_positions'):
        args.max_source_positions = args.max_positions
        args.max_target_positions = args.max_positions
    if not hasattr(args, 'share_input_output_embed'):
        args.share_input_output_embed = False
    return args


def maybe_no_grad(condition=True):
    if hasattr(torch, 'no_grad') and condition:
        return torch.no_grad()
    # no-op context manager
    return contextlib.ExitStack()


def volatile_variable(*args, **kwargs):
    if hasattr(torch, 'no_grad'):
        # volatile has been deprecated, use the no_grad context manager instead
        return Variable(*args, **kwargs)
    else:
        return Variable(*args, **kwargs, volatile=True)


def make_variable(sample, volatile=False, cuda=False):
    """Wrap input tensors in Variable class."""

    if len(sample) == 0:
        return {}

    def _make_variable(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            if cuda and torch.cuda.is_available():
                maybe_tensor = maybe_tensor.cuda()
            if volatile:
                return volatile_variable(maybe_tensor)
            else:
                return Variable(maybe_tensor)
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


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str):
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, 'r') as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
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


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def convert_padding_direction(
    src_tokens,
    src_lengths,
    padding_idx,
    right_to_left=False,
    left_to_right=False,
):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if pad_mask.max() == 0:
        # no padding, return early
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)

def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor
