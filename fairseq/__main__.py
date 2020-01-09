#!/usr/bin/env python

import sys
import random

import torch

from fairseq import options, distributed_utils
from fairseq.cli import eval_lm, generate, interactive
from fairseq.cli import preprocess, score, train, validate


def eval_lm_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    eval_lm.main(args)


def generate_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    generate.main(args)


def interactive_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    interactive.main(args)


def preprocess_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    preprocess.main(args)


def score_main():
    parser = score.get_parser()
    args = parser.parse_args()
    score.main(args)


def train_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=train.distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            train.distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=train.distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        train.main(args)


def validate_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    validate.main(args, override_args)    

if __name__ == "__main__":
    cli_mapping = {
        "eval-lm": eval_lm_main,
        "generate": generate_main,
        "interactive": interactive_main,
        "preprocess": preprocess_main,
        "score": score_main,
        "train": train_main,
        "validate": validate_main,
    }

    if len(sys.argv) == 1:
        print("Available commands:", ", ".join(cli_mapping))
    command = sys.argv.pop(1)
    if command in cli_mapping:
        cli_mapping[command]()
    else:
        available = ", ".join(cli_mapping)
        print("Unknown command: {}\nAvailable: {}".format(command, available), file=sys.stderr)
