# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import logging
import os
import torch
import traceback

from torch.autograd import Variable
from torch.serialization import default_restore_location

from fairseq import criterions, data, models


def parse_args_and_arch(parser):
    args = parser.parse_args()
    args.model = models.arch_model_map[args.arch]
    args = getattr(models, args.model).parse_arch(args)
    return args


def build_model(args, dataset):
    assert hasattr(models, args.model), 'Missing model type'
    return getattr(models, args.model).build_model(args, dataset)


def build_criterion(args, dataset):
    padding_idx = dataset.dst_dict.pad()
    if args.label_smoothing > 0:
        return criterions.LabelSmoothedCrossEntropyCriterion(args.label_smoothing, padding_idx)
    else:
        return criterions.CrossEntropyCriterion(padding_idx)


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
                'optimizer': optimizer.state_dict(),
                'best_loss': lr_scheduler.best,
            }
        ],
        'extra_state': extra_state,
    }
    torch_persistent_save(state_dict, filename)


def load_state(filename, model, criterion, optimizer, lr_scheduler, cuda_device=None):
    if not os.path.exists(filename):
        return None, []
    if cuda_device is None:
        state = torch.load(filename)
    else:
        state = torch.load(
            filename,
            map_location=lambda s, l: default_restore_location(s, 'cuda:{}'.format(cuda_device))
        )
    state = _upgrade_state_dict(state)

    # load model parameters
    model.load_state_dict(state['model'])

    # only load optimizer and lr_scheduler if they match with the checkpoint
    optim_history = state['optimizer_history']
    last_optim = optim_history[-1]
    if last_optim['criterion_name'] == criterion.__class__.__name__:
        optimizer.load_state_dict(last_optim['optimizer'])
        lr_scheduler.best = last_optim['best_loss']

    return state['extra_state'], optim_history


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    # add optimizer_history
    if 'optimizer_history' not in state:
        state['optimizer_history'] = [
            {
                'criterion_name': criterions.CrossEntropyCriterion.__name__,
                'optimizer': state['optimizer'],
                'best_loss': state['best_loss'],
            },
        ]
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
    return state


def load_ensemble_for_inference(filenames, data_path, split):
    # load model architectures and weights
    states = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        states.append(
            torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        )

    # load dataset
    args = states[0]['args']
    dataset = data.load(data_path, [split], args.source_lang, args.target_lang)

    # build models
    ensemble = []
    for state in states:
        model = build_model(args, dataset)
        model.load_state_dict(state['model'])
        ensemble.append(model)

    return ensemble, dataset


def prepare_sample(sample, volatile=False, cuda_device=None):
    """Wrap input tensors in Variable class."""

    def make_variable(tensor):
        if cuda_device is not None and torch.cuda.is_available():
            tensor = tensor.cuda(async=True, device=cuda_device)
        return Variable(tensor, volatile=volatile)

    return {
        'id': sample['id'],
        'ntokens': sample['ntokens'],
        'target': make_variable(sample['target']),
        'net_input': {
            key: make_variable(sample[key])
            for key in ['src_tokens', 'src_positions', 'input_tokens', 'input_positions']
        },
    }
