# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import collections
import os
import torch
import math

from fairseq import bleu, data, options, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.multiprocessing_trainer import MultiprocessingTrainer
from fairseq.progress_bar import progress_bar
from fairseq.sequence_generator import SequenceGenerator


def main():
    parser = options.get_parser('Trainer')
    dataset_args = options.add_dataset_args(parser)
    dataset_args.add_argument('--max-tokens', default=6000, type=int, metavar='N',
                              help='maximum number of tokens in a batch')
    dataset_args.add_argument('--train-subset', default='train', metavar='SPLIT',
                              choices=['train', 'valid', 'test'],
                              help='data subset to use for training (train, valid, test)')
    dataset_args.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                              help='comma separated list ofdata subsets '
                                   ' to use for validation (train, valid, valid1,test, test1)')
    dataset_args.add_argument('--test-subset', default='test', metavar='SPLIT',
                              help='comma separated list ofdata subset '
                                   'to use for testing (train, valid, test)')
    options.add_optimization_args(parser)
    options.add_checkpoint_args(parser)
    options.add_model_args(parser)

    args = utils.parse_args_and_arch(parser)
    print(args)

    if args.no_progress_bar:
        progress_bar.enabled = False
        progress_bar.print_interval = args.log_interval

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.manual_seed(args.seed)

    # Load dataset
    dataset = data.load_with_check(args.data, args.source_lang, args.target_lang)
    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.source_lang, args.target_lang = dataset.src, dataset.dst

    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in dataset.splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    num_gpus = torch.cuda.device_count()

    print('| using {} GPUs (with max tokens per GPU = {})'.format(num_gpus, args.max_tokens))

    # Build model
    print('| model {}'.format(args.arch))
    model = utils.build_model(args, dataset)
    criterion = utils.build_criterion(args, dataset)

    # Start multiprocessing
    trainer = MultiprocessingTrainer(args, model)

    # Load the latest checkpoint if one is available
    epoch, batch_offset = trainer.load_checkpoint(os.path.join(args.save_dir, args.restore_file))

    # Train until the learning rate gets too small
    val_loss = None
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > args.min_lr and epoch <= max_epoch:
        # train for one epoch
        train(args, epoch, batch_offset, trainer, criterion, dataset, num_gpus)

        # evaluate on validate set
        for k, subset in enumerate(args.valid_subset.split(',')):
            val_loss = validate(args, epoch, trainer, criterion, dataset, subset, num_gpus)
            if k == 0:
                if not args.no_save:
                    # save checkpoint
                    trainer.save_checkpoint(args, epoch, 0, val_loss)
                # only use first validation loss to update the learning schedule
                lr = trainer.lr_step(val_loss, epoch)

        epoch += 1
        batch_offset = 0
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))

    # Generate on test set and compute BLEU score
    for beam in [1, 5, 10, 20]:
        for subset in args.test_subset.split(','):
            scorer = score_test(args, trainer.get_model(), dataset, subset, beam,
                                cuda_device=(0 if num_gpus > 0 else None))
            print('| Test on {} with beam={}: {}'.format(subset, beam, scorer.result_string()))

    # Stop multiprocessing
    trainer.stop()


def train(args, epoch, batch_offset, trainer, criterion, dataset, num_gpus):
    """Train the model for one epoch."""

    itr = dataset.dataloader(args.train_subset, num_workers=args.workers,
                             max_tokens=args.max_tokens, seed=args.seed, epoch=epoch,
                             max_positions=args.max_positions,
                             sample_without_replacement=args.sample_without_replacement)
    loss_meter = AverageMeter()
    bsz_meter = AverageMeter()    # sentences per batch
    wpb_meter = AverageMeter()    # words per batch
    wps_meter = TimeMeter()       # words per second
    clip_meter = AverageMeter()   # % of updates clipped
    gnorm_meter = AverageMeter()  # gradient norm

    desc = '| epoch {:03d}'.format(epoch)
    lr = trainer.get_lr()
    with progress_bar(itr, desc, leave=False) as t:
        for i, sample in data.skip_group_enumerator(t, num_gpus, batch_offset):
            loss, grad_norm = trainer.train_step(sample, criterion)

            ntokens = sum(s['ntokens'] for s in sample)
            src_size = sum(s['src_tokens'].size(0) for s in sample)
            loss_meter.update(loss, ntokens)
            bsz_meter.update(src_size)
            wpb_meter.update(ntokens)
            wps_meter.update(ntokens)
            clip_meter.update(1 if grad_norm > args.clip_norm else 0)
            gnorm_meter.update(grad_norm)

            t.set_postfix(collections.OrderedDict([
                ('loss', '{:.2f} ({:.2f})'.format(loss, loss_meter.avg)),
                ('wps', '{:5d}'.format(round(wps_meter.avg))),
                ('wpb', '{:5d}'.format(round(wpb_meter.avg))),
                ('bsz', '{:5d}'.format(round(bsz_meter.avg))),
                ('lr', lr),
                ('clip', '{:3.0f}%'.format(clip_meter.avg * 100)),
                ('gnorm', '{:.4f}'.format(gnorm_meter.avg)),
            ]), refresh=False)

            if i == 0:
                # ignore the first mini-batch in words-per-second calculation
                wps_meter.reset()
            if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
                trainer.save_checkpoint(args, epoch, i + 1)

        fmt = desc + ' | train loss {:2.2f} | train ppl {:3.2f}'
        fmt += ' | s/checkpoint {:7d} | words/s {:6d} | words/batch {:6d}'
        fmt += ' | bsz {:5d} | lr {:0.6f} | clip {:3.0f}% | gnorm {:.4f}'
        t.write(fmt.format(loss_meter.avg, math.pow(2, loss_meter.avg),
                           round(wps_meter.elapsed_time),
                           round(wps_meter.avg),
                           round(wpb_meter.avg),
                           round(bsz_meter.avg),
                           lr, clip_meter.avg * 100,
                           gnorm_meter.avg))


def validate(args, epoch, trainer, criterion, dataset, subset, ngpus):
    """Evaluate the model on the validation set and return the average loss."""

    itr = dataset.dataloader(subset, batch_size=None,
                             max_tokens=args.max_tokens,
                             max_positions=args.max_positions)
    loss_meter = AverageMeter()

    desc = '| epoch {:03d} | valid on \'{}\' subset'.format(epoch, subset)
    with progress_bar(itr, desc, leave=False) as t:
        for _, sample in data.skip_group_enumerator(t, ngpus):
            ntokens = sum(s['ntokens'] for s in sample)
            loss = trainer.valid_step(sample, criterion)
            loss_meter.update(loss, ntokens)
            t.set_postfix(loss='{:.2f}'.format(loss_meter.avg), refresh=False)

        val_loss = loss_meter.avg
        t.write(desc + ' | valid loss {:2.2f} | valid ppl {:3.2f}'
                .format(val_loss, math.pow(2, val_loss)))

    # update and return the learning rate
    return val_loss


def score_test(args, model, dataset, subset, beam, cuda_device):
    """Evaluate the model on the test set and return the BLEU scorer."""

    translator = SequenceGenerator([model], dataset.dst_dict, beam_size=beam)
    if torch.cuda.is_available():
        translator.cuda()

    scorer = bleu.Scorer(dataset.dst_dict.pad(), dataset.dst_dict.eos(), dataset.dst_dict.unk())
    itr = dataset.dataloader(subset, batch_size=4, max_positions=args.max_positions)
    for _, _, ref, hypos in translator.generate_batched_itr(itr, cuda_device=cuda_device):
        scorer.add(ref.int().cpu(), hypos[0]['tokens'].int().cpu())
    return scorer


if __name__ == '__main__':
    main()
