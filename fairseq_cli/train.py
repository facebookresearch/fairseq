#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

from itertools import chain
import logging
import math
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch

from fairseq import (
    checkpoint_utils, data, distributed_utils, metrics, optim, options,
    progress_bar, tasks, utils
)
from fairseq.data import data_utils, iterators
from fairseq.trainer import Trainer
from fairseq.meters import StopwatchMeter


import pytorch_lightning as pl


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


#class DistributedDataset(fairseq.data.BaseWrapperDataset):
#
#    def __init__(self, dataset, num_replicas):
#        super().__init__(dataset)
#        if dataset.supports_prefetch:
#            # TODO
#            from fairseq import pdb; pdb.set_trace()
#            #self.prefetch([0])
#            pass
#        self.num_replicas = num_replicas
#        self.len = num_replicas * int(math.ceil(len(dataset) / float(num_replicas)))
#
#    def __getitem__(self, index):
#        if index >= len(self.dataset):
#            return {
#                'dummy': True,
#                # TODO this needs to be prefetched...
#                'item': self.dataset[0],
#            }
#        else:
#            return {
#                'dummy': False,
#                'item': self.dataset[index],
#            }
#        if index > self.max_index:
#            dummy_flag =
#
#    def __len__(self):
#        return self.len
#
#    def __iter__(self):
#        # deterministically shuffle based on epoch
#        g = torch.Generator()
#        g.manual_seed(self.epoch)
#        if self.shuffle:
#            indices = torch.randperm(len(self.dataset), generator=g).tolist()
#        else:
#            indices = list(range(len(self.dataset)))
#
#
#        # add extra samples to make it evenly divisible
#        indices += indices[:(self.total_size - len(indices))]
#        assert len(indices) == self.total_size
#
#        # subsample
#        indices = indices[self.rank:self.total_size:self.num_replicas]
#        assert len(indices) == self.num_samples
#
#        return iter(indices)
#
#    def __len__(self):
#        return self.num_samples
#
#    def set_epoch(self, epoch):
#        self.epoch = epoch
#        self.dataset.set_epoch(epoch)


class FairseqLightningModule(pl.LightningModule):

    def __init__(self, args, task, model, criterion):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model
        self.criterion = criterion

        self._dummy_batch = None

    def forward(self, batch):
        return self.criterion(self.model, batch)

    def training_step(self, batch, batch_idx):
        if self._dummy_batch is None:
            self._dummy_batch = batch

        self._set_seed()

        loss, sample_size, logging_output = self.forward(batch)
        logging_outputs = [logging_output]

        if self.use_ddp or self.use_ddp2:
            logging_outputs, sample_size = self._all_gather_list_sync(
                logging_outputs, sample_size
            )

        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)

        return {
            "loss": loss,
            "log": logging_output,
        }

    def configure_optimizers(self):
        return optim.build_optimizer(self.args, self.parameters()).optimizer

    @pl.data_loader
    def train_dataloader(self):
        self.task.load_dataset(
            self.args.train_subset,
            epoch=self.current_epoch,
            combine=True,
            # TODO data_selector=data_selector,
        )
        return self._get_dataloader(
            dataset=self.task.dataset(self.args.train_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.args.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=1,
            shard_id=0,
            num_workers=self.args.num_workers,
        )

    @pl.data_loader
    def val_dataloader(self):
        return self._get_dataloader(
            dataset=self.task.dataset(self.args.valid_subset.split(',')[0]),
            max_tokens=self.args.max_tokens_valid,
            max_sentences=self.args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=1,
            shard_id=0,
            num_workers=self.args.num_workers,
        )

    def _get_dataloader(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0,
        shuffle=False,
    ):
        # TODO assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(self.current_epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        batch_sampler = list(data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        ))

        # prefetch data for datasets that require this
        # TODO this isn't aware of per-GPU shards
        if getattr(dataset, 'supports_prefetch', False):
            dataset.prefetch([i for s in batch_sampler for i in s])

        #def shuffle_batches(batches, seed):
        #    with data_utils.numpy_seed(seed):
        #        np.random.shuffle(batches)
        #    return batches

        #else:
        #    if shuffle:
        #        batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
        #    else:
        #        batches = self.frozen_batches
        #    batches = list(ShardedIterator(
        #        batches, self.num_shards, self.shard_id, fill_value=[]
        #    ))

        if num_workers > 0:
            os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            # TODO num_workers=self.args.num_workers,
            num_workers=0,
        )

    def _set_seed(self):
        seed = self.args.seed + self.global_step
        torch.manual_seed(seed)
        if self.on_gpu:
            torch.cuda.manual_seed(seed)

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        results = list(zip(
            *distributed_utils.all_gather_list(
                [logging_outputs] + list(extra_stats_to_sum),
                max_size=getattr(self.args, 'all_gather_list_size', 16384),
            )
        ))
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return [logging_outputs] + extra_stats_to_sum

    def _reduce_and_log_stats(self, logging_outputs, sample_size):
        with metrics.aggregate() as agg:
            # convert logging_outputs to CPU to avoid unnecessary
            # device-to-host transfers in reduce_metrics
            logging_outputs = utils.apply_to_sample(
                lambda t: t.to(device='cpu', non_blocking=True),
                logging_outputs
            )

            self.task.reduce_metrics(logging_outputs, self.criterion)

            # support legacy interface
            logging_output = agg.get_smoothed_values()
            logging_output["sample_size"] = sample_size
            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]
            return logging_output


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #if init_distributed:
    #    args.distributed_rank = distributed_utils.distributed_init(args)

    #if distributed_utils.is_master(args):
    #    checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    #from fairseq import pdb; pdb.set_trace()
    trainer = pl.Trainer(
        gpus=args.distributed_world_size,
        distributed_backend='ddp',
        # TODO
    )
    pl_model = FairseqLightningModule(args, task, model, criterion)
    trainer.fit(pl_model)

    #trainer = Trainer(args, task, model, criterion)
    #logger.info('training on {} GPUs'.format(args.distributed_world_size))
    #logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
    #    args.max_tokens,
    #    args.max_sentences,
    #))

    ## Load the latest checkpoint if one is available and restore the
    ## corresponding train iterator
    #extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    ## Train until the learning rate gets too small
    #max_epoch = args.max_epoch or math.inf
    #max_update = args.max_update or math.inf
    #lr = trainer.get_lr()
    #train_meter = StopwatchMeter()
    #train_meter.start()
    #valid_subsets = args.valid_subset.split(',')
    #while (
    #    lr > args.min_lr
    #    and (
    #        epoch_itr.epoch < max_epoch
    #        # allow resuming training from the final checkpoint
    #        or epoch_itr._next_epoch_itr is not None
    #    )
    #    and trainer.get_num_updates() < max_update
    #):
    #    # train for one epoch
    #    train(args, trainer, task, epoch_itr)

    #    if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
    #        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
    #    else:
    #        valid_losses = [None]

    #    # only use first validation loss to update the learning rate
    #    lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

    #    # save checkpoint
    #    if epoch_itr.epoch % args.save_interval == 0:
    #        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    #    # early stop
    #    if should_stop_early(args, valid_losses[0]):
    #        logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
    #        break

    #    epoch_itr = trainer.get_train_iterator(
    #        epoch_itr.epoch,
    #        # sharded data: get train iterator for next epoch
    #        load_dataset=(os.pathsep in getattr(args, 'data', '')),
    #    )
    #train_meter.stop()
    #logger.info('done training in {:.1f} seconds'.format(train_meter.sum))


def should_stop_early(args, valid_loss):
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        return should_stop_early.num_runs > args.patience


@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for samples in progress:
        log_output = trainer.train_step(samples)
        num_updates = trainer.get_num_updates()
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(metrics.get_smoothed_values('train'))
        progress.log(stats, tag='train', step=num_updates)

        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    #if args.distributed_init_method is None:
    #    distributed_utils.infer_init_method(args)

    #if args.distributed_init_method is not None:
    #    # distributed training
    #    if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
    #        start_rank = args.distributed_rank
    #        args.distributed_rank = None  # assign automatically
    #        torch.multiprocessing.spawn(
    #            fn=distributed_main,
    #            args=(args, start_rank),
    #            nprocs=torch.cuda.device_count(),
    #        )
    #    else:
    #        distributed_main(args.device_id, args)
    #elif args.distributed_world_size > 1:
    #    # fallback for single node with multiple GPUs
    #    assert args.distributed_world_size <= torch.cuda.device_count()
    #    port = random.randint(10000, 20000)
    #    args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
    #    args.distributed_rank = None  # set based on device id
    #    if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
    #        logger.info('NOTE: you may get faster training with: --ddp-backend=no_c10d')
    #    torch.multiprocessing.spawn(
    #        fn=distributed_main,
    #        args=(args, ),
    #        nprocs=args.distributed_world_size,
    #    )
    #else:
        # single GPU training
    main(args)


if __name__ == '__main__':
    cli_main()
