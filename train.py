#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import collections
import os
import sys
import torch
import math
import torch.distributed
import torch.cuda


from fairseq import criterions, data, models, options, progress_bar
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.trainer import Trainer


def main(args=None):
    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')

    if not args:
        args = parse_train_args()
        args.device_id = 0

    os.makedirs(args.save_dir, exist_ok=True)

    if args.max_sentences_valid is None:
        args.max_sentences_valid = args.max_sentences

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

    print(args)
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    torch.cuda.set_device(args.device_id)
    # Build model and criterion
    model = models.build_model(args, dataset.src_dict, dataset.dst_dict)
    criterion = criterions.build_criterion(args, dataset.src_dict, dataset.dst_dict)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.data.numel() for p in model.parameters())))

    # The max number of positions can be different for train and valid
    # e.g., RNNs may support more positions at test time than seen in training
    max_positions_train = (
        min(args.max_source_positions, model.max_encoder_positions()),
        min(args.max_target_positions, model.max_decoder_positions())
    )
    max_positions_valid = (model.max_encoder_positions(), model.max_decoder_positions())

    gpus_str = '{} GPUs'.format(args.distributed_world_size)
    print('| using {} (with max tokens per GPU = {} and max sentences per GPU = {})'.format(
        gpus_str, args.max_tokens, args.max_sentences))

    # Start multiprocessing
    trainer = Trainer(args, model, criterion)

    # Load the latest checkpoint if one is available
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    extra_state = trainer.load_checkpoint(checkpoint_path)
    if extra_state is not None:
        epoch = extra_state['epoch']
        batch_offset = extra_state['batch_offset']
        print('| loaded checkpoint {} (epoch {})'.format(checkpoint_path, epoch))
        if batch_offset == 0:
            trainer.lr_step(epoch)
            epoch += 1
    else:
        epoch, batch_offset = 1, 0

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > args.min_lr and epoch <= max_epoch:
        # train for one epoch
        train(args, epoch, batch_offset, trainer, dataset, max_positions_train)

        # evaluate on validate set
        for k, subset in enumerate(args.valid_subset.split(',')):
            val_loss = validate(args, epoch, trainer, dataset,
                                max_positions_valid, subset)
            if k == 0:
                # only use first validation loss to update the learning schedule
                lr = trainer.lr_step(epoch, val_loss)

                if not args.no_save:
                    # save checkpoint
                    save_checkpoint(trainer, args, epoch, 0, val_loss)

        epoch += 1
        batch_offset = 0
    train_meter.stop()

    print('| done training in {:.1f} seconds'.format(train_meter.sum))

def parse_train_args():
    parser = options.get_parser('Trainer')
    options.add_dataset_args(parser, train=True)
    options.add_optimization_args(parser)
    options.add_checkpoint_args(parser)
    options.add_model_args(parser)
    options.add_distributed_training_args(parser)
    args = options.parse_args_and_arch(parser)
    return args


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
        args.train_subset,
        max_tokens=args.max_tokens, max_sentences=args.max_sentences,
        max_positions=max_positions, seed=seed, epoch=epoch,
        sample_without_replacement=args.sample_without_replacement,
        sort_by_source_size=(epoch <= args.curriculum),
        shard_id=args.distributed_rank, num_shards=args.distributed_world_size,
    )
    loss_meter = AverageMeter()
    nll_loss_meter = AverageMeter()
    bsz_meter = AverageMeter()    # sentences per batch
    wpb_meter = AverageMeter()    # words per batch
    wps_meter = TimeMeter()       # words per second
    clip_meter = AverageMeter()   # % of updates clipped
    extra_meters = collections.defaultdict(lambda: AverageMeter())

    lr = None
    num_updates = trainer.get_num_updates()
    with progress_bar.build_progress_bar(args, itr, epoch, no_progress_bar='simple') as t:
        for i, sample in enumerate(
            data.skip_group_enumerator(t, batch_offset),
            start=num_updates,
        ):
            loss_dict = trainer.train_step(sample)
            loss = loss_dict['loss']
            lr = loss_dict['lr']
            ntokens = loss_dict['ntokens']
            nsentences = loss_dict['nsentences']
            del loss_dict['loss']  # don't include in extra_meters or extra_postfix
            del loss_dict['lr']
            del loss_dict['ntokens']
            del loss_dict['nsentences']

            if 'nll_loss' in loss_dict:
                nll_loss = loss_dict['nll_loss']
                nll_loss_meter.update(nll_loss, ntokens)

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
                ('num_updates', i),
                ('lr', lr),
                ('clip', '{:.0%}'.format(clip_meter.avg)),
            ] + extra_postfix))

            if i == 0:
                # ignore the first mini-batch in words-per-second calculation
                wps_meter.reset()
            if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
                save_checkpoint(trainer, args, epoch, i + 1, 0)

        t.print(collections.OrderedDict([
            ('train loss', round(loss_meter.avg, 2)),
            ('train ppl', get_perplexity(nll_loss_meter.avg
                                         if nll_loss_meter.count > 0
                                         else loss_meter.avg)),
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
    elif not args.no_epoch_checkpoints:
        epoch_filename = os.path.join(
            args.save_dir, 'checkpoint{}_{}.pt'.format(epoch, batch_offset))
        trainer.save_checkpoint(epoch_filename, extra_state)

    last_filename = os.path.join(args.save_dir, 'checkpoint_last.pt')
    trainer.save_checkpoint(last_filename, extra_state)


def validate(args, epoch, trainer, dataset, max_positions, subset):
    """Evaluate the model on the validation set and return the average loss."""

    itr = dataset.eval_dataloader(
        subset, max_tokens=args.max_tokens, max_sentences=args.max_sentences_valid,
        max_positions=max_positions,
        skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
        descending=True,  # largest batch first to warm the caching allocator
        shard_id=args.distributed_rank, num_shards=args.distributed_world_size,
    )
    loss_meter = AverageMeter()
    nll_loss_meter = AverageMeter()
    extra_meters = collections.defaultdict(lambda: AverageMeter())

    prefix = 'valid on \'{}\' subset'.format(subset)
    with progress_bar.build_progress_bar(args, itr, epoch, prefix, no_progress_bar='simple') as t:
        for sample in data.skip_group_enumerator(t):
            loss_dict = trainer.valid_step(sample)
            ntokens = loss_dict['ntokens']
            loss = loss_dict['loss']
            del loss_dict['loss']  # don't include in extra_meters or extra_postfix
            del loss_dict['ntokens']
            del loss_dict['nsentences']

            if 'nll_loss' in loss_dict:
                nll_loss = loss_dict['nll_loss']
                nll_loss_meter.update(nll_loss, ntokens)

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
            ('valid ppl', get_perplexity(nll_loss_meter.avg
                                         if nll_loss_meter.count > 0
                                         else loss_meter.avg)),
        ] + [
            (k, meter.avg)
            for k, meter in extra_meters.items()
        ]))

    # update and return the learning rate
    return loss_meter.avg


if __name__ == '__main__':
    main()
