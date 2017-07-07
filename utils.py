import os
import torch
from torch.autograd import Variable

import models


def build_model(args, dataset):
    if args.arch == 'fconv':
        encoder_layers = eval(args.encoder_layers)
        decoder_layers = eval(args.decoder_layers)
        decoder_attention = eval(args.decoder_attention)
        model = models.fconv(
            dataset, args.dropout, args.encoder_embed_dim, encoder_layers,
            args.decoder_embed_dim, decoder_layers, decoder_attention)
    else:
        model = models.__dict__[args.arch](dataset, args.dropout)
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


def save_checkpoint(save_dir, epoch, batch_offset, model, optimizer, lr_scheduler, val_loss=None):
    state_dict = {
        'epoch': epoch,
        'batch_offset': batch_offset,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': lr_scheduler.best,
        'val_loss': val_loss,
    }

    if batch_offset == 0:
        epoch_filename = os.path.join(save_dir, f'checkpoint{epoch}.pt')
        torch_persistent_save(state_dict, epoch_filename)

        assert val_loss is not None
        if not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best:
            save_checkpoint.best = val_loss
            best_filename = os.path.join(save_dir, 'checkpoint_best.pt')
            torch_persistent_save(state_dict, best_filename)

    last_filename = os.path.join(save_dir, 'checkpoint_last.pt')
    torch_persistent_save(state_dict, last_filename)


def load_checkpoint(filename, model, optimizer=None, lr_scheduler=None):
    if not os.path.exists(filename):
        return 0, 0

    state = torch.load(filename)
    model.load_state_dict(state['model'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    if lr_scheduler:
        lr_scheduler.best = state['best_loss']
    epoch = state['epoch']
    batch_offset = state['batch_offset']

    print('| loaded checkpoint {} (epoch {})'.format(filename, epoch))
    return epoch, batch_offset


def prepare_sample(sample, volatile=False, use_cuda=True):
    """Wrap input tensors in Variable class"""
    r = {'id': sample['id'], 'ntokens': sample['ntokens']}
    for key in ['src_tokens', 'src_positions', 'input_tokens', 'input_positions', 'target']:
        tensor = sample[key]
        if use_cuda and torch.cuda.is_available():
            tensor = tensor.cuda(async=True)
        r[key] = Variable(tensor, volatile=volatile)
    return r
