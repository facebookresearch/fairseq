#!/usr/bin/env python3
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

from fairseq import data, options, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.multiprocessing_trainer import MultiprocessingTrainer


def main():
    parser = options.get_parser('Trainer')
    dataset_args = options.add_dataset_args(parser)
    dataset_args.add_argument('--max-tokens', default=6000, type=int, metavar='N',
                              help='maximum number of tokens in a batch')
    dataset_args.add_argument('--max-sentences', type=int, metavar='N',
                              help='maximum number of sentences in a batch')
    dataset_args.add_argument('--train-subset', default='train', metavar='SPLIT',
                              choices=['train', 'valid', 'test'],
                              help='data subset to use for training (train, valid, test)')
    dataset_args.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                              help='comma separated list of data subsets '
                                   ' to use for validation (train, valid, valid1,test, test1)')
    options.add_optimization_args(parser)
    options.add_checkpoint_args(parser)
    options.add_model_args(parser)

    args = utils.parse_args_and_arch(parser)

    if args.no_progress_bar and args.log_format is None:
        args.log_format = 'simple'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.manual_seed(args.seed)

    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        dataset = data.load_dataset(args.data, splits, args.source_lang, args.target_lang)
    else:
        dataset = data.load_raw_text_dataset(args.data, splits, args.source_lang, args.target_lang)
    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.source_lang, args.target_lang = dataset.src, dataset.dst

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    args.num_gpus = torch.cuda.device_count()

    print(args)
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    print('| using {} GPUs (with max tokens per GPU = {} and max sentences per GPU = {})'.format(
        args.num_gpus, args.max_tokens, args.max_sentences))

    # Build model and criterion
    model = utils.build_model(args, dataset.src_dict, dataset.dst_dict)
    criterion = utils.build_criterion(args, dataset.src_dict, dataset.dst_dict)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.data.numel() for p in model.parameters())))

    # The max number of positions can be different for train and valid
    # e.g., RNNs may support more positions at test time than seen in training
    max_positions_train = (args.max_source_positions, args.max_target_positions)
    max_positions_valid = (
        min(args.max_source_positions, model.max_encoder_positions()),
        min(args.max_target_positions, model.max_decoder_positions())
    )

    # Start multiprocessing
    trainer = MultiprocessingTrainer(args, model, criterion)

    # Load the latest checkpoint if one is available
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    extra_state = trainer.load_checkpoint(checkpoint_path)
    if extra_state is not None:
        epoch = extra_state['epoch']
        batch_offset = extra_state['batch_offset']
        print('| loaded checkpoint {} (epoch {})'.format(checkpoint_path, epoch))
        if batch_offset == 0:
            epoch += 1
    else:
        epoch, batch_offset = 1, 0

    # Train until the learning rate gets too small
    val_loss = None
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > args.min_lr and epoch <= max_epoch:
        # train for one epoch
        train(args, epoch, batch_offset, trainer, dataset, max_positions_train)

        # evaluate on validate set
        for k, subset in enumerate(args.valid_subset.split(',')):
            val_loss = validate(args, epoch, trainer, dataset, max_positions_valid, subset)
            if k == 0:
                if not args.no_save:
                    # save checkpoint
                    save_checkpoint(trainer, args, epoch, 0, val_loss)
                # only use first validation loss to update the learning schedule
                lr = trainer.lr_step(val_loss, epoch)

        epoch += 1
        batch_offset = 0
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))

    # Stop multiprocessing
    trainer.stop()


def get_perplexity(loss):
    try:
        return round(math.pow(2, loss), 2)
    except OverflowError:
        return float('inf')


def train(args, epoch, batch_offset, trainer, dataset, max_positions):
    """Train the model for one epoch."""

    seed = args.seed + epoch
    torch.manual_seed(seed)
    trainer.set_seed(seed)

    itr = dataset.train_dataloader(
        args.train_subset, num_workers=args.workers,
        max_tokens=args.max_tokens, max_sentences=args.max_sentences,
        max_positions=max_positions, seed=seed, epoch=epoch,
        sample_without_replacement=args.sample_without_replacement,
        sort_by_source_size=(epoch <= args.curriculum))
    loss_meter = AverageMeter()
    bsz_meter = AverageMeter()    # sentences per batch
    wpb_meter = AverageMeter()    # words per batch
    wps_meter = TimeMeter()       # words per second
    clip_meter = AverageMeter()   # % of updates clipped
    extra_meters = collections.defaultdict(lambda: AverageMeter())

    lr = trainer.get_lr()
    with utils.build_progress_bar(args, itr, epoch) as t:
        for i, sample in data.skip_group_enumerator(t, args.num_gpus, batch_offset):
            loss_dict = trainer.train_step(sample)
            loss = loss_dict['loss']
            del loss_dict['loss']  # don't include in extra_meters or extra_postfix

            ntokens = sum(s['ntokens'] for s in sample)
            nsentences = sum(s['net_input']['src_tokens'].size(0) for s in sample)
            loss_meter.update(loss, nsentences if args.sentence_avg else ntokens)
            bsz_meter.update(nsentences)
            wpb_meter.update(ntokens)
            wps_meter.update(ntokens)
            clip_meter.update(1 if loss_dict['gnorm'] > args.clip_norm else 0)

            extra_postfix = []
            for k, v in loss_dict.items():
                extra_meters[k].update(v)
                extra_postfix.append((k, extra_meters[k].avg))

            t.log(collections.OrderedDict([
                ('loss', loss_meter),
                ('wps', round(wps_meter.avg)),
                ('wpb', round(wpb_meter.avg)),
                ('bsz', round(bsz_meter.avg)),
                ('lr', lr),
                ('clip', '{:.0%}'.format(clip_meter.avg)),
            ] + extra_postfix))

            if i == 0:
                # ignore the first mini-batch in words-per-second calculation
                wps_meter.reset()
            if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
                save_checkpoint(trainer, args, epoch, i + 1)

        t.print(collections.OrderedDict([
            ('train loss', round(loss_meter.avg, 2)),
            ('train ppl', get_perplexity(loss_meter.avg)),
            ('s/checkpoint', round(wps_meter.elapsed_time)),
            ('words/s', round(wps_meter.avg)),
            ('words/batch', round(wpb_meter.avg)),
            ('bsz', round(bsz_meter.avg)),
            ('lr', lr),
            ('clip', '{:3.0f}%'.format(clip_meter.avg * 100)),
        ] + [
            (k, meter.avg)
            for k, meter in extra_meters.items()
        ]))


def save_checkpoint(trainer, args, epoch, batch_offset, val_loss):
    extra_state = {
        'epoch': epoch,
        'batch_offset': batch_offset,
        'val_loss': val_loss,
    }

    if batch_offset == 0:
        if not args.no_epoch_checkpoints:
            epoch_filename = os.path.join(args.save_dir, 'checkpoint{}.pt'.format(epoch))
            trainer.save_checkpoint(epoch_filename, extra_state)

        assert val_loss is not None
        if not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best:
            save_checkpoint.best = val_loss
            best_filename = os.path.join(args.save_dir, 'checkpoint_best.pt')
            trainer.save_checkpoint(best_filename, extra_state)

    last_filename = os.path.join(args.save_dir, 'checkpoint_last.pt')
    trainer.save_checkpoint(last_filename, extra_state)


def validate(args, epoch, trainer, dataset, max_positions, subset):
    """Evaluate the model on the validation set and return the average loss."""

    itr = dataset.eval_dataloader(
        subset, max_tokens=args.max_tokens, max_sentences=args.max_sentences,
        max_positions=max_positions,
        skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
        descending=True,  # largest batch first to warm the caching allocator
    )
    loss_meter = AverageMeter()
    extra_meters = collections.defaultdict(lambda: AverageMeter())

    prefix = 'valid on \'{}\' subset'.format(subset)
    with utils.build_progress_bar(args, itr, epoch, prefix) as t:
        for _, sample in data.skip_group_enumerator(t, args.num_gpus):
            loss_dict = trainer.valid_step(sample)
            loss = loss_dict['loss']
            del loss_dict['loss']  # don't include in extra_meters or extra_postfix

            ntokens = sum(s['ntokens'] for s in sample)
            loss_meter.update(loss, ntokens)

            extra_postfix = []
            for k, v in loss_dict.items():
                extra_meters[k].update(v)
                extra_postfix.append((k, extra_meters[k].avg))

            t.log(collections.OrderedDict([
                ('valid loss', round(loss_meter.avg, 2)),
            ] + extra_postfix))

        t.print(collections.OrderedDict([
            ('valid loss', round(loss_meter.avg, 2)),
            ('valid ppl', get_perplexity(loss_meter.avg)),
        ] + [
            (k, meter.avg)
            for k, meter in extra_meters.items()
        ]))

    # update and return the learning rate
    return loss_meter.avg


if __name__ == '__main__':
    main()
