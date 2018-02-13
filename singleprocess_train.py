#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import collections
import itertools
import os
import math
import torch

from fairseq import criterions, data, models, options, progress_bar
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.trainer import Trainer


def main(args):
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        dataset = data.load_dataset(
            args.data, splits, args.source_lang, args.target_lang)
    else:
        dataset = data.load_raw_text_dataset(
            args.data, splits, args.source_lang, args.target_lang)
    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.source_lang, args.target_lang = dataset.src, dataset.dst
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    # Build model and criterion
    model = models.build_model(args, dataset.src_dict, dataset.dst_dict)
    criterion = criterions.build_criterion(args, dataset.src_dict, dataset.dst_dict)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.data.numel() for p in model.parameters())))

    # Build trainer
    trainer = Trainer(args, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available
    os.makedirs(args.save_dir, exist_ok=True)
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
        train(args, trainer, dataset, epoch, batch_offset)

        # evaluate on validate set
        for k, subset in enumerate(args.valid_subset.split(',')):
            val_loss = validate(args, trainer, dataset, subset, epoch)
            if k == 0:
                # only use first validation loss to update the learning schedule
                lr = trainer.lr_step(epoch, val_loss)

                # save checkpoint
                if not args.no_save:
                    save_checkpoint(trainer, args, epoch, 0, val_loss)

        epoch += 1
        batch_offset = 0
    train_meter.stop()

    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, dataset, epoch, batch_offset):
    """Train the model for one epoch."""

    # Set seed based on args.seed and the epoch number so that we get
    # reproducible results when resuming from checkpoints
    seed = args.seed + epoch
    torch.manual_seed(seed)

    # The max number of positions can be different for train and valid
    # e.g., RNNs may support more positions at test time than seen in training
    max_positions_train = (
        min(args.max_source_positions, trainer.get_model().max_encoder_positions()),
        min(args.max_target_positions, trainer.get_model().max_decoder_positions())
    )

    # Initialize dataloader, starting at batch_offset
    itr = dataset.train_dataloader(
        args.train_subset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions_train,
        seed=seed,
        epoch=epoch,
        sample_without_replacement=args.sample_without_replacement,
        sort_by_source_size=(epoch <= args.curriculum),
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )
    progress = progress_bar.build_progress_bar(args, itr, epoch, no_progress_bar='simple')
    itr = itertools.islice(progress, batch_offset, None)

    # reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for i, sample in enumerate(itr, start=batch_offset):
        log_output = trainer.train_step(sample)

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss']:
                continue  # these are already logged above
            extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # save mid-epoch checkpoints
        if i == batch_offset:
            # ignore the first mini-batch in words-per-second calculation
            trainer.get_meter('wps').reset()
        if args.save_interval > 0 and trainer.get_num_updates() % args.save_interval == 0:
            save_checkpoint(trainer, args, epoch, i + 1)

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    return stats


def validate(args, trainer, dataset, subset, epoch):
    """Evaluate the model on the validation set and return the average loss."""

    # Initialize dataloader
    max_positions_valid = (
        trainer.get_model().max_encoder_positions(),
        trainer.get_model().max_decoder_positions(),
    )
    itr = dataset.eval_dataloader(
        subset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=max_positions_valid,
        skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
        descending=True,  # largest batch first to warm the caching allocator
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )
    progress = progress_bar.build_progress_bar(
        args, itr, epoch,
        prefix='valid on \'{}\' subset'.format(subset),
        no_progress_bar='simple'
    )

    # reset validation loss meters
    for k in ['valid_loss', 'valid_nll_loss']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for sample in progress:
        log_output = trainer.valid_step(sample)

        # log mid-validation stats
        stats = get_valid_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss']:
                continue
            extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

    # log validation stats
    stats = get_valid_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    return stats['valid_loss']


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(trainer, args, epoch, batch_offset, val_loss=None):
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


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
