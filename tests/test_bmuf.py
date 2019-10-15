import argparse
import random
import unittest
from multiprocessing import Manager

import torch
import torch.nn as nn
from fairseq import distributed_utils, optim


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        return output


def setup_model_loss_criterion(args, rank, is_cuda):
    """
    setup model, criterion and optimizer based on input args
    """
    args.distributed_rank = rank
    distributed_utils.distributed_init(args)
    torch.manual_seed(1)
    model = Model(args.input_size, args.nb_classes)
    loss_fn = nn.CrossEntropyLoss()
    if is_cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    optimizer = optim.sgd.SGD(args, model.parameters())
    optimizer = optim.FairseqBMUF(args, optimizer)

    return model, loss_fn, optimizer


def train_step(input, target, model, loss_fn, optimizer):
    """Do forward, backward and parameter update."""
    model.train()
    output = model(input)
    loss = loss_fn(output, target)
    optimizer.backward(loss)
    optimizer.step()


def single_gpu_training(args, rank, iterations, shared_results):

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.set_device(rank)

    model, loss_fn, optimizer = setup_model_loss_criterion(args, rank, is_cuda)

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
    args.distributed_backend = "gloo"

    args.distributed_world_size = 2
    port = random.randint(10000, 20000)
    args.distributed_init_method = "tcp://localhost:{port}".format(port=port)
    args.distributed_init_host = "localhost"
    args.distributed_port = port + 1
    args.local_world_size = args.distributed_world_size
    return args


class TestBMUF(unittest.TestCase):
    def bmuf_process(self, args, iterations):
        processes = []
        results = Manager().dict()
        ctx = torch.multiprocessing.get_context("spawn")
        for rank in range(args.distributed_world_size):
            p = ctx.Process(
                target=single_gpu_training, args=(args, rank, iterations, results)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Make sure params in both machines are same
        assert len(results) == 2
        self.assertAlmostEqual(results[0], results[1])

    def test_bmuf_sync(self):
        # Train model for 1 iteration and do bmuf sync without doing warmup
        args = setup_args()
        iterations = 1
        self.bmuf_process(args, iterations)

    def test_warmup_sync(self):
        # Train model for 20 iteration and do warmup sync without doing bmuf sync
        args = setup_args()
        args.warmup_iterations = 20
        iterations = 20
        self.bmuf_process(args, iterations)

    def test_warmup_sync_bmuf_sync(self):
        # Train model for 25 iteration and do warmup sync after 20 iteration
        # and bmuf sync after 25 iteration
        args = setup_args()
        args.warmup_iterations = 20
        args.global_sync_iter = 5
        iterations = 25
        self.bmuf_process(args, iterations)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)
