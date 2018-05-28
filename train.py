#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import collections
import os
import math
import torch

from itertools import islice

from fairseq import criterions, models, options, progress_bar
from fairseq.data import data_utils, data_loaders
from fairseq.fp16_trainer import FP16Trainer
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.utils import checkpoint_paths


def main(args):
    if args.max_tokens is None:
        args.max_tokens = 6000

    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Load dataset
    splits = ['train', 'valid']
    dataset = load_dataset(args, splits)
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
    if args.fp16:
        trainer = FP16Trainer(args, model, criterion)
    else:
        if torch.cuda.get_device_capability(0)[0] >= 7:
            print('| NOTICE: your device may support faster training with --fp16')
        trainer = Trainer(args, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize dataloader
    train_dataloader = dataset.train_dataloader_generator(
        args.train_subset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=(
            min(args.max_source_positions, trainer.get_model().max_encoder_positions()),
            min(args.max_target_positions, trainer.get_model().max_decoder_positions())
        ),
        seed=args.seed,
        sample_without_replacement=args.sample_without_replacement,
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )

    # Load the latest checkpoint if one is available
    epoch, next_ds = load_checkpoint(args, trainer, train_dataloader)

    # Send a dummy batch to warm the caching allocator
    dummy_batch = data_utils.get_dummy_batch(args.max_tokens, dataset.src_dict, dataset.dst_dict)
    trainer.dummy_train_step(dummy_batch)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    first_val_loss = None
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > args.min_lr and epoch <= max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, next_ds, epoch, dataset)

        if epoch % args.validate_interval == 0:
            first_val_loss = val_loss(args, trainer, dataset, epoch)

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch, first_val_loss)

        # save checkpoint
        if not args.no_save and epoch % args.save_interval == 0:
            save_checkpoint(trainer, args, epoch, end_of_epoch=True, val_loss=first_val_loss)

        epoch += 1
        next_ds = next(train_dataloader)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def load_dataset(args, splits):
    is_raw = not data_utils.has_binary_files(args.data, splits)
    dataset = data_loaders.load_dataset(args, splits, is_raw)
    return dataset


def train(args, trainer, itr, epoch, dataset):
    """Train the model for one epoch."""

    # Set seed based on args.seed and the epoch number so that we get
    # reproducible results when resuming from checkpoints
    seed = args.seed + epoch
    torch.manual_seed(seed)

    # reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    # update parameters every N batches
    if epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    max_update = args.max_update or math.inf
    num_batches = len(itr)
    progress = progress_bar.build_progress_bar(args, itr, epoch, no_progress_bar='simple')
    for i, sample in enumerate(progress):
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            # buffer updates according to --update-freq
            trainer.train_step(sample, update_params=False)
            continue
        else:
            log_output = trainer.train_step(sample, update_params=True)

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if not args.no_save and (args.save_interval_updates or 0) > 0 and num_updates % args.save_interval_updates == 0:
            first_val_loss = val_loss(args, trainer, dataset, epoch, num_updates)
            save_checkpoint(trainer, args, epoch, end_of_epoch=False, val_loss=first_val_loss)

        if num_updates >= max_update:
            break

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
    stats['oom'] = trainer.get_meter('oom').avg
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale').avg)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    return stats


def validate(args, trainer, dataset, subset, epoch, num_updates):
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

    # log validation stats
    stats = get_valid_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg

    if num_updates is not None:
        stats['num_updates'] = num_updates

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


def val_loss(args, trainer, dataset, epoch, num_updates=None):
    # evaluate on validate set
    subsets = args.valid_subset.split(',')
    # we want to validate all subsets so the results get printed out, but return only the first
    losses = [validate(args, trainer, dataset, subset, epoch, num_updates) for subset in subsets]
    return losses[0] if len(losses) > 0 else None


def save_checkpoint(trainer, args, epoch, end_of_epoch, val_loss):
    extra_state = {
        'epoch': epoch,
        'val_loss': val_loss,
        'wall_time': trainer.get_meter('wall').elapsed_time,
        'end_of_epoch': end_of_epoch,
    }

    if end_of_epoch and not args.no_epoch_checkpoints:
        epoch_filename = os.path.join(args.save_dir, 'checkpoint{}.pt'.format(epoch))
        trainer.save_checkpoint(epoch_filename, extra_state)
    elif not end_of_epoch and args.keep_interval_updates > 0:
        checkpoint_filename = os.path.join(args.save_dir,
                                           'checkpoint_{}_{}.pt'.format(epoch, trainer.get_num_updates()))
        trainer.save_checkpoint(checkpoint_filename, extra_state)
        # remove old checkpoints
        checkpoints = checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        # checkpoints are sorted in descending order
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)

    assert val_loss is not None
    if not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best:
        save_checkpoint.best = val_loss
        best_filename = os.path.join(args.save_dir, 'checkpoint_best.pt')
        trainer.save_checkpoint(best_filename, extra_state)

    last_filename = os.path.join(args.save_dir, 'checkpoint_last.pt')
    trainer.save_checkpoint(last_filename, extra_state)


def load_checkpoint(args, trainer, train_dataloader):
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    epoch = 1
    ds = None
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            epoch = extra_state['epoch']
            end_of_epoch = extra_state.get('end_of_epoch', True)
            trainer_updates = trainer.get_num_updates()

            print('| loaded checkpoint {} (epoch {})'.format(checkpoint_path, epoch))

            trainer.lr_step(epoch)
            updates = 0
            for i in range(epoch):
                ds = next(train_dataloader)
                updates += len(ds)

            if not end_of_epoch and ds is not None and updates > trainer_updates:
                completed_batches = len(ds) - (updates - trainer_updates)
                assert completed_batches >= 0
                ds = iter(ds)

                print('| resuming from batch {}'.format(completed_batches + 1))

                # consume completed batches
                next(islice(ds, completed_batches, completed_batches), None)
            else:
                if not end_of_epoch:
                    print('| WARNING: checkpoint is not at end of epoch')
                ds = next(train_dataloader)
                epoch += 1

            trainer.get_meter('wall').reset(init=extra_state.get('wall_time', 0))
    return epoch, ds or next(train_dataloader)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
