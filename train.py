import argparse
import os
import torch
import math
from progress_bar import progress_bar

import bleu
import data
import generate
import models
import utils
from average_meter import AverageMeter, TimeMeter
from multiprocessing_trainer import MultiprocessingTrainer


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
parser.add_argument('--max-tokens', default=6000, type=int, metavar='N',
                    help='maximum number of tokens in a batch')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')

# optimization
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--min-lr', metavar='LR', default=1e-5, type=float,
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
parser.add_argument('--restore-file', default='checkpoint_last.pt',
                    help='filename in save-dir from which to load checkpoint')
parser.add_argument('--save-interval', type=int, default=-1,
                    help='checkpoint every this many batches')
parser.add_argument('--no-progress-bar', action='store_true',
                    help='disable progress bar')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='log progress every N updates (when progress bar is disabled)')
parser.add_argument('--seed', default=1, type=int, metavar='N',
                    help='pseudo random number generator seed')

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
parser.add_argument('--decoder-out-embed-dim', default=256, type=int, metavar='N',
                    help='decoder output embedding dimension')

def main():
    global args
    args = parser.parse_args()
    print(args)

    if args.no_progress_bar:
        progress_bar.enabled = False
        progress_bar.print_interval = args.log_interval

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.manual_seed(args.seed)

    dataset = data.load(args.data, args.source_lang, args.target_lang)
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in ['train', 'valid', 'test']:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    num_gpus = torch.cuda.device_count()
    args.max_tokens *= num_gpus
    print('| using {} GPUs (with max tokens = {})'.format(num_gpus, args.max_tokens))

    print('| model {}'.format(args.arch))
    model = utils.build_model(args, dataset)

    # Start multiprocessing
    trainer = MultiprocessingTrainer(args, model)

    # Load the latest checkpoint if one is available
    epoch, batch_offset = trainer.load_checkpoint(
        os.path.join(args.save_dir, args.restore_file))

    # Train until the learning rate gets too small
    val_loss = None
    lr = trainer.get_lr()
    while lr > args.min_lr:
        # train for one epoch
        train(epoch, batch_offset, trainer, dataset)

        # evaluate on validate set
        val_loss, lr = validate(epoch, trainer, dataset)

        # save checkpoint
        trainer.save_checkpoint(args.save_dir, epoch, 0, val_loss)

        epoch += 1
        batch_offset = 0

    # Generate on test set and compute BLEU score
    for beam in [1, 5, 10, 20]:
        scorer = score_test(epoch, trainer.get_model(), dataset, beam)
        print('| Test with beam={}: BLEU4 = {:2.2f}'.format(beam, scorer.score()))

    # Stop multiprocessing
    trainer.stop()


def train(epoch, batch_offset, trainer, dataset):
    """Train the model for one epoch"""

    itr = dataset.dataloader('train',
                             num_workers=args.workers,
                             max_tokens=args.max_tokens,
                             seed=(args.seed, epoch))
    loss_meter = AverageMeter()
    spb_meter = AverageMeter()  # sentences per batch
    wpb_meter = AverageMeter()  # words per batch
    wps_meter = TimeMeter()     # words per second

    desc = '| epoch {}'.format(epoch)
    lr = trainer.get_lr()
    with progress_bar(itr, desc, leave=False) as t:
        for i, sample in enumerate(t):
            if i < batch_offset:
                continue
            loss = trainer.train_step(sample)

            loss_meter.update(loss, sample['ntokens'])
            spb_meter.update(sample['src_tokens'].size(0))
            wpb_meter.update(sample['ntokens'])
            wps_meter.update(sample['ntokens'])

            t.set_postfix(loss='{:.2f} ({:.2f})'.format(loss, loss_meter.avg),
                          spb='{:5d}'.format(round(spb_meter.avg)),
                          wpb='{:5d}'.format(round(wpb_meter.avg)),
                          wps='{:5d}'.format(round(wps_meter.avg)),
                          lr=lr)

            if i == 0:
                # ignore the first mini-batch in words-per-second calculation
                wps_meter.reset()
            if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
                trainer.save_checkpoint(args.save_dir, epoch, i + 1)

        t.write('| epoch {:03d} | train loss {:2.2f} | train ppl {:3.2f} | words/s {:6d} | lr {:0.6f}'
                .format(epoch, loss_meter.avg, math.pow(2, loss_meter.avg),
                        round(wps_meter.avg), lr))


def validate(epoch, trainer, dataset):
    """Evaluate the model on the validation set and return the average loss"""

    itr = dataset.dataloader('valid', batch_size=None, max_tokens=args.max_tokens)
    loss_meter = AverageMeter()

    desc = '| val {}'.format(epoch)
    with progress_bar(itr, desc, leave=False) as t:
        for sample in t:
            loss = trainer.valid_step(sample)
            loss_meter.update(loss, sample['ntokens'])
            t.set_postfix(loss='{:.2f}'.format(loss_meter.avg))

        val_loss = loss_meter.avg
        t.write('| epoch {:03d} | val loss {:2.2f} | val ppl {:3.2f}'
                .format(epoch, val_loss, math.pow(2, val_loss)))

    # update and return the learning rate
    return val_loss, trainer.lr_step(val_loss, epoch)


def score_test(epoch, model, dataset, beam):
    """Evaluate the model on the test set and print the BLEU score"""
    translator = generate.SequenceGenerator(model, dataset.dst_dict, beam_size=beam)
    if torch.cuda.is_available():
        translator.cuda()

    scorer = bleu.Scorer(dataset.dst_dict.pad(), dataset.dst_dict.eos())
    itr = dataset.dataloader('test', batch_size=4)
    for id, src, ref, hypos in generate.generate_batched_itr(translator, itr):
        scorer.add(ref.int().cpu(), hypos[0]['tokens'].int().cpu())
    return scorer


if __name__ == '__main__':
    main()
