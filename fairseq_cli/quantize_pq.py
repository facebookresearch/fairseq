#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Quantize an already trained model with iterative PQ on one or across multiple GPUs.
This works best if you trained your model with quantization noise.
"""

import logging
import math
import os
import sys
import yaml

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, train_utils, utils
from fairseq.logging import meters
from fairseq.trainer import Trainer
from fairseq.modules.quantization.pq import quantize_model_, SizeTracker
from fairseq.modules.quantization.quantization_options import parse_config_yaml


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.quantize_pq')


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
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # quantize model

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    logger.info('training on {} GPUs'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Parse config yaml file
    if args.quantization_config_path:
        with open(args.quantization_config_path) as config_file:
            config = parse_config_yaml(yaml.safe_load(config_file))
    else:
        config = parse_config_yaml({})

    n_centroids_config = config["n_centroids"]
    block_sizes_config = config["block_sizes"]
    layers_to_quantize = config["layers_to_quantize"]

    size_tracker = SizeTracker(model)

    # Quantize model by stages
    for step in range(len(layers_to_quantize)):

        # quantize model inplace
        quantized_layers = quantize_model_(
                        model,
                        size_tracker,
                        layers_to_quantize,
                        block_sizes_config,
                        n_centroids_config,
                        step=step,
                    )
        logger.info(f"Finetuning stage {step}, quantized layers: {quantized_layers}")
        logger.info(f"{size_tracker}")

        # Re-create trainer since model parameters have changed
        trainer = Trainer(args, task, model, criterion)

        # Train until the learning rate gets too small
        max_epoch = args.max_epoch or math.inf
        max_update = args.max_update or math.inf
        lr = trainer.get_lr()
        train_meter = meters.StopwatchMeter()
        train_meter.start()
        valid_subsets = args.valid_subset.split(',')

        # finetune centroids
        while trainer.get_num_updates() < max_update:
            # train for one epoch
            train_utils.train(args, trainer, task, epoch_itr)

            if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
                valid_losses = train_utils.validate(args, trainer, task, epoch_itr, valid_subsets)
            else:
                valid_losses = [None]

            # only use first validation loss to update the learning rate
            lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

            # save checkpoint
            if epoch_itr.epoch % args.save_interval == 0:
                checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

            # early stop
            if train_utils.should_stop_early(args, valid_losses[0]):
                logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
                break

            epoch_itr = trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                # sharded data: get train iterator for next epoch
                load_dataset=(os.pathsep in getattr(args, 'data', '')),
            )
        train_meter.stop()
        logger.info('done training in {:.1f} seconds'.format(train_meter.sum))


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    # quantization configuration path is only used for iterative PQ quantization
    group = parser.add_argument_group('Quantization')
    group.add_argument('--quantization-config-path', default=None,
                       help='Path to Quantization Config File')
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    train_utils.call_main(args, main, modify_parser)


if __name__ == '__main__':
    cli_main()
