# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import functools
import random
import unittest
from multiprocessing import Manager

import torch
import torch.nn as nn
from fairseq import optim
from fairseq.distributed import utils as distributed_utils
from omegaconf import OmegaConf


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        return output


def setup_model_loss_criterion(cfg, args, rank, is_cuda):
    """
    setup model, criterion and optimizer based on input args
    """
    args.distributed_rank = rank
    cfg.distributed_training.distributed_rank = args.distributed_rank
    if cfg.distributed_training.distributed_world_size > 1:
        distributed_utils.distributed_init(cfg)
    torch.manual_seed(1)
    model = Model(args.input_size, args.nb_classes)
    loss_fn = nn.CrossEntropyLoss()
    if is_cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    optimizer = optim.sgd.SGD(args, model.parameters())
    optimizer = optim.FairseqBMUF(
        cfg=cfg.bmuf,
        optimizer=optimizer
    )

    return model, loss_fn, optimizer


def train_step(input, target, model, loss_fn, optimizer, **unused):
    """Do forward, backward and parameter update."""
    model.train()
    output = model(input)
    loss = loss_fn(output, target)
    optimizer.backward(loss)
    optimizer.step()


def single_gpu_training(cfg, args, rank, iterations, shared_results):

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.set_device(rank)

    model, loss_fn, optimizer = setup_model_loss_criterion(cfg, args, rank, is_cuda)

    for _ in range(iterations):
        input = torch.randn(1, args.input_size)
        target = torch.empty(args.batch_size, dtype=torch.long).random_(args.nb_classes)

        if is_cuda:
            input = input.cuda()
            target = target.cuda()
        train_step(input, target, model, loss_fn, optimizer)

    results = []
    for param in model.parameters():
        if len(results) == 0:
            results = param.flatten().cpu().data
        else:
            results = torch.cat((results, param.flatten().cpu().data), 0)

    shared_results[rank] = results


def setup_args():
    args = argparse.Namespace()
    args.global_sync_iter = 20
    args.block_momentum = 0.875
    args.block_lr = 0.5
    args.input_size = 5
    args.nb_classes = 2
    args.batch_size = 1
    args.lr = [1e-3]
    args.momentum = 0
    args.weight_decay = 0
    args.warmup_iterations = 0
    args.use_nbm = True
    args.average_sync = True
    args.global_sync_iter = 1
    args.model_parallel_size = 1
    args.distributed_backend = "gloo"

    args.distributed_world_size = 2
    port = random.randint(10000, 20000)
    args.distributed_init_method = "tcp://localhost:{port}".format(port=port)
    args.distributed_init_host = "localhost"
    args.distributed_port = port + 1
    args.local_world_size = args.distributed_world_size

    cfg = OmegaConf.create()
    cfg.optimization = OmegaConf.create()
    cfg.common = OmegaConf.create()
    cfg.distributed_training = OmegaConf.create()
    cfg.dataset = OmegaConf.create()
    cfg.bmuf = OmegaConf.create()
    cfg.optimizer = OmegaConf.create()

    cfg.bmuf.global_sync_iter = args.global_sync_iter
    cfg.bmuf.block_momentum = args.block_momentum
    cfg.bmuf.block_lr = args.block_lr
    cfg.dataset.batch_size = args.batch_size
    cfg.optimization.lr = args.lr
    cfg.optimizer.momentum = args.momentum
    cfg.optimizer.weight_decay = args.weight_decay
    cfg.bmuf.warmup_iterations = args.warmup_iterations
    cfg.bmuf.use_nbm = args.use_nbm
    cfg.bmuf.average_sync = args.average_sync
    cfg.common.model_parallel_size = args.model_parallel_size
    cfg.distributed_training.distributed_backend = args.distributed_backend
    cfg.distributed_training.distributed_world_size = args.distributed_world_size
    cfg.bmuf.distributed_world_size = args.distributed_world_size
    cfg.distributed_training.distributed_init_method = args.distributed_init_method
    cfg.distributed_training.distributed_port = args.distributed_port

    return cfg, args


@unittest.skipIf(torch.cuda.device_count() < 2, "test requires 2 GPUs")
class TestBMUF(unittest.TestCase):
    def bmuf_process(self, cfg, args, iterations):
        processes = []
        results = Manager().dict()
        torch.multiprocessing.spawn(
            fn=functools.partial(single_gpu_training, cfg, args),
            args=(iterations, results),
            nprocs=args.distributed_world_size,
            join=True,
        )
        return results

    def test_bmuf_sync(self):
        # Train model for 1 iteration and do bmuf sync without doing warmup
        cfg, args = setup_args()
        iterations = 1
        results = self.bmuf_process(cfg, args, iterations)
        # Make sure params in both machines are same
        assert len(results) == 2
        self.assertAlmostEqual(results[0], results[1])

    def test_warmup_sync(self):
        # Train model for 20 iteration and do warmup sync without doing bmuf sync
        cfg, args = setup_args()
        args.warmup_iterations = 20
        cfg.bmuf.warmup_iterations = args.warmup_iterations
        iterations = 20
        results = self.bmuf_process(cfg, args, iterations)
        # Make sure params in both machines are same
        assert len(results) == 2
        self.assertAlmostEqual(results[0], results[1])

    def test_warmup_sync_bmuf_sync(self):
        # Train model for 25 iteration and do warmup sync after 20 iteration
        # and bmuf sync after 25 iteration
        cfg, args = setup_args()
        args.warmup_iterations = 20
        args.global_sync_iter = 5
        cfg.bmuf.warmup_iterations = args.warmup_iterations
        cfg.bmuf.global_sync_iter = args.global_sync_iter
        iterations = 25
        results = self.bmuf_process(cfg, args, iterations)
        # Make sure params in both machines are same
        assert len(results) == 2
        self.assertAlmostEqual(results[0], results[1])

    def test_single_gpu_bmuf(self):
        # Train model for 5 iterations and use GPU 1
        cfg, args = setup_args()
        args.distributed_world_size = 1
        args.warmup_iterations = 5
        cfg.distributed_training.distributed_world_size = args.distributed_world_size
        cfg.bmuf.distributed_world_size = args.distributed_world_size
        cfg.bmuf.warmup_iterations = args.warmup_iterations
        iterations = 20
        results = self.bmuf_process(cfg, args, iterations)
        assert len(results) == 1

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
