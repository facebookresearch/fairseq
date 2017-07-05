import argparse
import os
import torch
import math
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from progress_bar import progress_bar

import models
import data
import generate
from nag import NAG
from average_meter import AverageMeter, TimeMeter


parser = argparse.ArgumentParser(description='Convolutional Sequence to Sequence Training')
parser.add_argument('data', metavar='DIR',
                    help='path to data directory')
parser.add_argument('--arch', '-a', default='fconv', metavar='ARCH',
                    choices=models.__all__,
                    help='model architecture ({})'.format(', '.join(models.__all__)))

# dataset and data loading
parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                    help='source language')
parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                    help='target language')
parser.add_argument('--batch-size', '-b', default=32, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--max-tokens', default=None, type=int, metavar='N',
                    help='maximum number of tokens in a batch')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')

# optimization
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--min-lr', metavar='LR', default=1e-4, type=float,
                    help='minimum learning rate')
parser.add_argument('--force-anneal', '--fa', default=0, type=int, metavar='N',
                    help='force annealing at specified epoch')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum factor')
parser.add_argument('--clip-norm', default=25, type=float, metavar='NORM',
                    help='clip threshold of gradients')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                    help='weight decay')
parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                    help='dropout probability')

# checkpointing and utilities
parser.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                    help='path to save checkpoints')
parser.add_argument('--save-file', default='checkpoint_last.pt',
                    help='filename in save-dir from which to load checkpoint')
parser.add_argument('--no-progress-bar', action='store_true',
                    help='disable progress bar')

# model configuration
parser.add_argument('--encoder-embed-dim', default=512, type=int, metavar='N',
                    help='encoder embedding dimension')
parser.add_argument('--encoder-layers', default='[(512, 3)] * 20', type=str, metavar='EXPR',
                    help='encoder layers [(dim, kernel_size), ...]')
parser.add_argument('--decoder-embed-dim', default=512, type=int, metavar='N',
                    help='decoder embedding dimension')
parser.add_argument('--decoder-layers', default='[(512, 3)] * 20', type=str, metavar='EXPR',
                    help='decoder layers [(dim, kernel_size), ...]')
parser.add_argument('--decoder-attention', default='True', type=str, metavar='EXPR',
                    help='decoder attention [True, ...]')


def main():
    global args
    args = parser.parse_args()
    print(args)

    if args.no_progress_bar:
        progress_bar.enabled = False

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = data.load(args.data, args.source_lang, args.target_lang)
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in ['train', 'valid', 'test']:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    print('| model {}'.format(args.arch))
    if args.arch == 'fconv':
        encoder_layers = eval(args.encoder_layers)
        decoder_layers = eval(args.decoder_layers)
        decoder_attention = eval(args.decoder_attention)
        model = models.fconv(
            dataset, args.dropout, args.encoder_embed_dim, encoder_layers,
            args.decoder_embed_dim, decoder_layers, decoder_attention)
    else:
        model = models.__dict__[args.arch](dataset, args.dropout)

    if torch.cuda.is_available():
        model.cuda()

    # Nesterov accelerated gradient
    optimizer = NAG(model.parameters(), args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    # Decay the LR by 0.1 every time the validation loss plateaus
    if args.force_anneal > 0:
        anneal = lambda e: 1 if e < args.force_anneal else 0.1 ** (e + 1 - args.force_anneal)
        lr_scheduler = LambdaLR(optimizer, anneal)
        lr_scheduler.best = None
    else:
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=0)

    # Load the latest checkpoint if one is available
    epoch = load_checkpoint(model, optimizer, lr_scheduler)

    # Train until the learning rate gets too small
    while optimizer.param_groups[0]['lr'] > args.min_lr:
        # train for one epoch
        train(epoch, model, dataset, optimizer)

        # evaluate on validate set
        val_loss = validate(epoch, model, dataset)

        # update the learning rate
        if args.force_anneal > 0:
            lr_scheduler.step(epoch + 1)
        else:
            lr_scheduler.step(val_loss, epoch + 1)

        epoch += 1

        # save checkpoint
        save_checkpoint(epoch, model, optimizer, lr_scheduler, val_loss)

    # Generate on test set and compute BLEU score
    scorer = generate.generate(model, dataset)
    print('| Test with beam=20: BLEU4 = {:2.2f}'.format(scorer.score()))


def train(epoch, model, dataset, optimizer):
    """Train the model for one epoch"""

    model.train()
    itr = dataset.dataloader('train', epoch=epoch, batch_size=args.batch_size,
                             num_workers=args.workers,
                             max_tokens=args.max_tokens)
    loss_meter = AverageMeter()
    wps_meter = TimeMeter()

    def step(sample):
        loss = model(*prepare_sample(sample))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_norm)
        optimizer.step()

        return loss.data[0] / math.log(2)

    desc = '| epoch {}'.format(epoch)
    with progress_bar(itr, desc, leave=False) as t:
        for i, sample in enumerate(t):
            loss = step(sample)

            loss_meter.update(loss, sample['ntokens'])
            wps_meter.update(sample['ntokens'])

            t.set_postfix(loss='{:.2f} ({:.2f})'.format(loss, loss_meter.avg),
                          wps='{:5d}'.format(round(wps_meter.avg)),
                          lr=optimizer.param_groups[0]['lr'])

            if i == 0:
                # ignore the first mini-batch in words-per-second calculation
                wps_meter.reset()

        t.write('| epoch {:03d} | train loss {:2.2f} | train ppl {:3.2f} | words/s {:6d} | lr {:0.6f}'
                .format(epoch, loss_meter.avg, math.pow(2, loss_meter.avg),
                        round(wps_meter.avg), optimizer.param_groups[0]['lr']))


def validate(epoch, model, dataset):
    """Evaluate the model on the validation set and return the average loss"""

    model.eval()
    itr = dataset.dataloader('valid', epoch=epoch, batch_size=args.batch_size)
    loss_meter = AverageMeter()

    def step(_sample):
        loss = model(*prepare_sample(sample, volatile=True))
        return loss.data[0] / math.log(2)

    desc = '| val {}'.format(epoch)
    with progress_bar(itr, desc, leave=False) as t:
        for sample in t:
            loss = step(sample)
            loss_meter.update(loss, sample['ntokens'])
            t.set_postfix(loss='{:.2f}'.format(loss_meter.avg))

        t.write('| epoch {:03d} | val loss {:2.2f} | val ppl {:3.2f}'
                .format(epoch, loss_meter.avg, math.pow(2, loss_meter.avg)))

    return loss_meter.avg


def save_checkpoint(epoch, model, optimizer, lr_scheduler, val_loss):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': lr_scheduler.best,
    }
    filename = os.path.join(args.save_dir, f'checkpoint{epoch}.pt')
    torch.save(state_dict, filename)

    if not hasattr(save_checkpoint, 'best') or val_loss <= save_checkpoint.best:
        save_checkpoint.best = val_loss
        best_filename = os.path.join(args.save_dir, 'checkpoint_best.pt')
        torch.save(state_dict, best_filename)

    last_filename = os.path.join(args.save_dir, 'checkpoint_last.pt')
    torch.save(state_dict, last_filename)

def load_checkpoint(model, optimizer, lr_scheduler):
    filename = os.path.join(args.save_dir, args.save_file)
    if not os.path.exists(filename):
        return 0

    state = torch.load(filename)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    lr_scheduler.best = state['best_loss']
    epoch = state['epoch']

    print('| loaded checkpoint {} (epoch {})'.format(filename, epoch))
    return epoch


def prepare_sample(sample, volatile=False):
    """Wrap input tensors in Variable class"""
    r = []
    for key in ['src_tokens', 'src_positions', 'input_tokens', 'input_positions', 'target']:
        tensor = sample[key]
        if torch.cuda.is_available():
            tensor = tensor.cuda(async=True)
        r.append(Variable(tensor, volatile=volatile))
    r.append(sample['ntokens'])
    return r


if __name__ == '__main__':
    main()
