# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import os
import torch
from torch.autograd import Variable
from torch.serialization import default_restore_location

from fairseq import models


def build_model(args, dataset):
    if args.arch == 'fconv':
        encoder_layers = eval(args.encoder_layers)
        decoder_layers = eval(args.decoder_layers)
        decoder_attention = eval(args.decoder_attention)
        model = models.fconv(
            dataset, args.dropout, args.encoder_embed_dim, encoder_layers,
            args.decoder_embed_dim, decoder_layers, decoder_attention,
            decoder_out_embed_dim=args.decoder_out_embed_dim,
            label_smoothing=args.label_smoothing)
    else:
        model = models.__dict__[args.arch](dataset, args.dropout,
                                           label_smoothing=args.label_smoothing)
    return model


def torch_persistent_save(*args, **kwargs):
    i = 1
    while True:
        try:
            return torch.save(*args, **kwargs)
        except:
            if i == 3:
                raise
            else:
                i += 1


def save_checkpoint(save_dir, epoch, batch_offset, model, optimizer, lr_scheduler, no_epoch_checkpoints, val_loss=None):
    state_dict = {
        'epoch': epoch,
        'batch_offset': batch_offset,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': lr_scheduler.best,
        'val_loss': val_loss,
    }

    if batch_offset == 0:
        if not no_epoch_checkpoints:
            epoch_filename = os.path.join(save_dir, 'checkpoint{}.pt'.format(epoch))
            torch_persistent_save(state_dict, epoch_filename)

        assert val_loss is not None
        if not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best:
            save_checkpoint.best = val_loss
            best_filename = os.path.join(save_dir, 'checkpoint_best.pt')
            torch_persistent_save(state_dict, best_filename)

    last_filename = os.path.join(save_dir, 'checkpoint_last.pt')
    torch_persistent_save(state_dict, last_filename)


def load_checkpoint(filename, model, optimizer=None, lr_scheduler=None, cuda_device=None):
    if not os.path.exists(filename):
        return 1, 0
    if cuda_device is None:
        state = torch.load(filename)
    else:
        state = torch.load(filename, map_location=lambda s,l:
            default_restore_location(s, 'cuda:{}'.format(cuda_device)))

    model.load_state_dict(state['model'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    if lr_scheduler:
        lr_scheduler.best = state['best_loss']
    epoch = state['epoch'] + 1
    batch_offset = state['batch_offset']

    gpu_str = ' on GPU #{}'.format(cuda_device) if cuda_device is not None else ''
    print('| loaded checkpoint {} (epoch {}){}'.format(filename, epoch, gpu_str))
    return epoch, batch_offset


def prepare_sample(sample, volatile=False, cuda_device=None):
    """Wrap input tensors in Variable class."""
    r = {
        'id': sample['id'],
        'ntokens': sample['ntokens'],
        'net_input': {},
    }
    # prepare input to network
    for key in ['src_tokens', 'src_positions', 'input_tokens', 'input_positions', 'target']:
        tensor = sample[key]
        if cuda_device is not None and torch.cuda.is_available():
            tensor = tensor.cuda(async=True, device=cuda_device)
        r['net_input'][key] = Variable(tensor, volatile=volatile)
    return r
