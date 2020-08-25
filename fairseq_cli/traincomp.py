#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import random
import sys
import os
import numpy as np
import torch

from fairseq import (
    checkpoint_utils, distributed_utils, options, tasks, utils
)
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import StopwatchMeter
from fairseq_cli.Comparable4 import Comparable

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

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
    trainer = Trainer(args, task, model, criterion)
    logger.info('training on {} GPUs'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    #extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
    if args.restore_file:
        extra_state, epoch = load_checkpoint(args, trainer)

    else:
        epoch = 1


    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    if max_epoch == math.inf: max_epoch = 5  # if args.max_epoch was not set appropriately, then set it to 2
    if args.comparable:
        # 1. Initialize Comparable object
        comp = Comparable(model, trainer, task, args)

        while epoch <= max_epoch and lr > args.min_lr and trainer.get_num_updates() < max_update:
            # 2. Update threshold if dynamic
            '''if args.threshold_dynamics != 'static' and epoch != 0:
                comp.update_threshold(args.threshold_dynamics, args.infer_threshold)
            '''
            comp.task.begin_epoch(epoch, comp.trainer.get_model())
            # 3. Extract parallel data and train
            comp.extract_and_train(args.comparable_data, epoch)

            if not args.disable_validation:
                valid_losses = comp.validate(epoch, valid_subsets)
            else:
                valid_losses = [None]

            # only use first validation loss to update the learning rate
            lr = comp.trainer.lr_step(epoch, valid_losses[0])
            #print('learning rate = ','{:.10f}'.format(lr), ' validation loss = ', '{:.10f}'.format(valid_losses[0] ))

            # 4.  save checkpoint
            comp.save_comp_chkp(epoch)
            epoch += 1

            # early stop
            if should_stop_early(args, valid_losses[0]):
                logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
                break
    train_meter.stop()
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))


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


def load_checkpoint(args, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """
    # only one worker should attempt to create the required dir
    if args.distributed_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.restore_file == "checkpoint_last.pt":
        checkpoint_path = os.path.join(args.save_dir, "checkpoint_last.pt")
    else:
        checkpoint_path = os.path.join(args.save_dir, args.restore_file)

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        args.reset_optimizer,
        args.reset_lr_scheduler,
        eval(args.optimizer_overrides),
        reset_meters=args.reset_meters,
    )

    if extra_state is not None and not args.reset_dataloader:
        # restore iterator from checkpoint
        epoch = extra_state["train_iterator"]["epoch"] + 1

    else:
        epoch = 1

    trainer.lr_step(epoch)

    return extra_state, epoch




def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            logger.info('NOTE: you may get faster training with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )

    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
