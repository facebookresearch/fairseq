# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import pickle
import random
import socket
import struct
import subprocess
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Mapping

import torch
import torch.distributed as dist

from fairseq import utils

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


logger = logging.getLogger(__name__)


# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Whether to use XLA ops (e.g., on TPUs) instead of CUDA ops.
_USE_XLA = False


def is_master(args):
    return args.distributed_rank == 0


def infer_init_method(args, force_distributed=False):
    if args.distributed_init_method is not None or getattr(args, 'tpu', False):
        return

    if args.pipeline_model_parallel:
        if args.pipeline_balance is None:
            raise ValueError('--pipeline-balance is currently required for pipeline model parallelism')
        if args.pipeline_devices is None:
            raise ValueError('--pipeline-devices is currently required for pipeline model parallelism')
        gpus_per_node = torch.cuda.device_count()
        num_pipeline_devices = len(set(args.pipeline_devices))
        assert gpus_per_node >= num_pipeline_devices and gpus_per_node % num_pipeline_devices == 0, (
            'the number of unique device IDs in --pipeline-devices must evenly divide '
            'the number of GPUs per node (multi-node pipelining is not yet supported)'
        )
        num_pipelines_per_node = gpus_per_node // num_pipeline_devices

    # support torch.distributed.launch
    if all(key in os.environ for key in [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK'
    ]):
        args.distributed_init_method = 'env://'
        args.distributed_world_size = int(os.environ['WORLD_SIZE'])
        args.distributed_rank = int(os.environ['RANK'])
        # processes are created by torch.distributed.launch
        args.distributed_no_spawn = True

    # we can determine the init method automatically for Slurm
    elif args.distributed_port > 0:
        node_list = os.environ.get('SLURM_STEP_NODELIST')
        if node_list is None:
            node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=args.distributed_port,
                )
                nnodes = int(os.environ.get('SLURM_NNODES'))
                ntasks_per_node = os.environ.get('SLURM_NTASKS_PER_NODE')
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get('SLURM_NTASKS'))
                    nnodes = int(os.environ.get('SLURM_NNODES'))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    gpus_per_node = torch.cuda.device_count()
                    node_id = int(os.environ.get('SLURM_NODEID'))
                    args.distributed_rank = node_id * gpus_per_node
                    args.distributed_world_size = nnodes * gpus_per_node
                elif args.pipeline_model_parallel:
                    assert ntasks_per_node == num_pipelines_per_node, (
                        'SLURM --ntasks-per-node must match number of pipelines per '
                        'node (={})'.format(num_pipelines_per_node)
                    )
                    args.distributed_no_spawn = True
                    # For 4-way MP on nodes with 8 GPUs, ranks will be [0, 1] on
                    # the first node, [1, 2] on the second node, etc. This
                    # matches torch.distributed.launch.
                    node_id = int(os.environ.get('SLURM_NODEID'))
                    local_id = int(os.environ.get('SLURM_LOCALID'))
                    args.distributed_rank = node_id * num_pipelines_per_node + local_id
                    # In the above example, device_id will always be in [0, 1],
                    # which also matches torch.distributed.launch.
                    args.device_id = local_id
                    # We also want to set distributed_world_size to be the total
                    # number of pipelines across all nodes.
                    args.distributed_world_size = nnodes * num_pipelines_per_node
                else:
                    assert ntasks_per_node == args.distributed_world_size // nnodes
                    args.distributed_no_spawn = True
                    args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                    args.device_id = int(os.environ.get('SLURM_LOCALID'))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass

    elif args.distributed_world_size > 1 or force_distributed:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)

    if args.pipeline_model_parallel:
        if not args.distributed_no_spawn:
            # When distributed_no_spawn is False, we expect distributed_rank and
            # distributed_world_size to be based on the total number of GPUs, so
            # we need to correct them to be based on the number of pipelines.
            assert args.distributed_world_size % num_pipeline_devices == 0
            args.distributed_world_size = args.distributed_world_size // num_pipeline_devices
            # In the case of 4-way MP on nodes with 8 GPUs, we want
            # distributed_rank to be the starting GPU index for each pipeline
            # i.e., 0, 2, ...
            assert args.distributed_rank % gpus_per_node == 0
            assert args.distributed_rank % num_pipeline_devices == 0
            args.distributed_rank = args.distributed_rank // num_pipeline_devices
            # launch one process per pipeline
            args.distributed_num_procs = num_pipelines_per_node

        # if we have 4-way MP on a node with 8 GPUs, we want device_ids to be 0
        # and 4, indicating the starting device IDs for each pipeline
        args.device_id *= num_pipeline_devices

        if args.device_id > 0:
            # if there's multiple pipelines on a node (e.g., 4-way MP on an 8
            # GPU node), we need to adjust pipeline_devices accordingly
            logger.debug(
                "setting CUDA device={} on rank {}"
                .format(args.device_id, args.distributed_rank)
            )
            torch.cuda.set_device(args.device_id)
            args.pipeline_devices = [args.device_id + d for d in args.pipeline_devices]
            logger.info(
                "setting pipeline_devices={} on rank {}"
                .format(args.pipeline_devices, args.distributed_rank),
            )
    elif not args.distributed_no_spawn:
        args.distributed_num_procs = min(
            torch.cuda.device_count(),
            args.distributed_world_size,
        )


def distributed_init(args):
    if not getattr(args, 'tpu', False):
        if torch.distributed.is_initialized():
            warnings.warn('Distributed is already initialized, cannot initialize twice!')
        else:
            logger.info('distributed init (rank {}): {}'.format(
                args.distributed_rank, args.distributed_init_method,
            ))
            dist.init_process_group(
                backend=args.distributed_backend,
                init_method=args.distributed_init_method,
                world_size=args.distributed_world_size,
                rank=args.distributed_rank,
            )
            logger.info('initialized host {} as rank {}'.format(
                socket.gethostname(), args.distributed_rank,
            ))

            # perform a dummy all-reduce to initialize the NCCL communicator
            if torch.cuda.is_available():
                dist.all_reduce(torch.zeros(1).cuda())

        args.distributed_rank = torch.distributed.get_rank()
    else:
        assert xm.xrt_world_size() == args.distributed_world_size
        global _USE_XLA
        _USE_XLA = True
        args.device_id = xm.get_local_ordinal()
        args.distributed_rank = xm.get_ordinal()
        xm.rendezvous('distributed_init')  # wait for all workers
        xm.mark_step()

    if not is_master(args):
        logging.getLogger().setLevel(logging.WARNING)

    # initialize data parallel and model parallel groups
    initialize_distributed_groups(model_parallel_size=args.model_parallel_size)

    # set torch seed
    utils.set_torch_seed(args.seed)

    # extra setup for model parallel
    if get_model_parallel_world_size() > 1:
        try:
            from fairseq.model_parallel.megatron import mpu
        except ImportError:
            raise ImportError(
                '\n\nPlease install the megatron submodule:'
                '\n\n  git submodule update --init '
                'fairseq/model_parallel/megatron'
            )
        model_part_number = get_model_parallel_rank()
        args.checkpoint_suffix += '-model_part-{0}'.format(model_part_number)

    return args.distributed_rank


def distributed_main(i, main, args, kwargs):
    args.device_id = i
    if torch.cuda.is_available() and not args.cpu and not getattr(args, "tpu", False):
        torch.cuda.set_device(args.device_id)
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = kwargs.pop('start_rank', 0) + i

    args.distributed_rank = distributed_init(args)

    after_distributed_init_fn = kwargs.pop('after_distributed_init_fn', None)
    if after_distributed_init_fn:
        args = after_distributed_init_fn(args)

    main(args, **kwargs)


def call_main(args, main, **kwargs):
    if args.distributed_init_method is None:
        infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            kwargs['start_rank'] = start_rank
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(main, args, kwargs),
                nprocs=args.distributed_num_procs,
            )
        else:
            distributed_main(args.device_id, main, args, kwargs)
    elif getattr(args, "tpu", False) and args.distributed_world_size > 1:
        import torch_xla.distributed.xla_multiprocessing as xmp
        torch.multiprocessing.set_sharing_strategy("file_system")
        xmp.spawn(
            fn=distributed_main,
            args=(main, args, kwargs),
            nprocs=8,  # use all 8 TPU cores
        )
    else:
        # single GPU main
        main(args, **kwargs)


def is_xla():
    global _USE_XLA
    return _USE_XLA


def get_rank(group):
    if is_xla():
        assert group[0] == 'tpu'
        my_group = _find_my_group(group[1])
        return my_group.index(get_global_rank())
    else:
        return dist.get_rank(group=group)


def get_world_size(group):
    if is_xla():
        assert group[0] == 'tpu'
        my_group = _find_my_group(group[1])
        return len(my_group)
    else:
        return dist.get_world_size(group=group)


def new_groups(grouped_ranks: List[List[int]]):
    if is_xla():
        return ('tpu', grouped_ranks)
    else:
        groups = [dist.new_group(g) for g in grouped_ranks]
        my_group_idx = _find_my_group_index(grouped_ranks)
        return groups[my_group_idx]


def all_reduce(tensor, group, op='sum'):
    if is_xla():
        assert isinstance(group, tuple) and group[0] == 'tpu'
        tensor = [tensor]  # wrap in a list to make xm.all_reduce in-place
        return xm.all_reduce(op, tensor, groups=group[1])[0]
    else:
        if op == 'sum':
            op = dist.ReduceOp.SUM
        elif op == 'max':
            op = dist.ReduceOp.MAX
        else:
            raise NotImplementedError
        dist.all_reduce(tensor, op=op, group=group)
        return tensor


def all_to_all(tensor, group):
    """Perform an all-to-all operation on a 1D Tensor."""
    assert tensor.dim() == 1
    split_count = get_world_size(group=group)
    assert tensor.numel() % split_count == 0
    if is_xla():
        assert isinstance(group, tuple) and group[0] == 'tpu'
        return xm.all_to_all(
            tensor,
            split_dimension=0,
            concat_dimension=0,
            split_count=split_count,
            groups=group[1],
        )
    else:
        output = torch.zeros_like(tensor)
        dist.all_to_all_single(output, tensor, group=group)
        return output


def all_gather(tensor, group, return_tensor=False):
    """Perform an all-gather operation."""
    if is_xla():
        result = xm.all_gather(tensor, groups=group[1])
        world_size = get_world_size(group=group)
        result = result.view(world_size, *tensor.size())
        if return_tensor:
            return result
        else:
            return [result[i] for i in range(world_size)]
    else:
        world_size = get_world_size(group=group)
        rank = get_rank(group=group)
        tensor_list = [
            tensor if i == rank else torch.empty_like(tensor)
            for i in range(world_size)
        ]
        dist.all_gather(tensor_list, tensor, group=group)
        if return_tensor:
            return torch.stack(tensor_list, dim=0)
        else:
            return tensor_list


def all_gather_list(data, group, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)

    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    data = utils.move_to_cpu(data)
    enc = pickle.dumps(data)
    enc_size = len(enc)
    header_size = 4  # size of header that contains the length of the encoded data
    size = header_size + enc_size
    if size > max_size:
        raise ValueError('encoded data size ({}) exceeds max_size ({})'.format(size, max_size))

    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = torch.ByteTensor(list(header + enc))
    start = rank * max_size
    buffer[start:start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size:(i + 1) * max_size]
            enc_size, = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(pickle.loads(bytes(out_buffer[header_size:header_size + enc_size].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data. '
            'Try rerunning with --ddp-backend=no_c10d and see if that helps.'
        )


def all_reduce_dict(
    data: Mapping[str, Any],
    device,
    group,
) -> Dict[str, Any]:
    """
    AllReduce a dictionary of values across workers. We separately
    reduce items that are already on the device and items on CPU for
    better performance.

    Args:
        data (Mapping[str, Any]): dictionary of data to all-reduce, but
            cannot be a nested dictionary
        device (torch.device): device for the reduction
        group: group of the collective
    """
    data_keys = list(data.keys())

    # We want to separately reduce items that are already on the
    # device and items on CPU for performance reasons.
    cpu_data = OrderedDict()
    device_data = OrderedDict()
    for k in data_keys:
        t = data[k]
        if not torch.is_tensor(t):
            cpu_data[k] = torch.tensor(t, dtype=torch.double)
        elif t.device.type != device.type:
            cpu_data[k] = t.to(dtype=torch.double)
        else:
            device_data[k] = t.to(dtype=torch.double)

    def _all_reduce_dict(data: OrderedDict):
        if len(data) == 0:
            return data
        buf = torch.stack(list(data.values())).to(device=device)
        all_reduce(buf, group=group)
        return {k: buf[i] for i, k in enumerate(data)}

    cpu_data = _all_reduce_dict(cpu_data)
    device_data = _all_reduce_dict(device_data)

    def get_from_stack(key):
        if key in cpu_data:
            return cpu_data[key]
        elif key in device_data:
            return device_data[key]
        raise KeyError

    return OrderedDict([(key, get_from_stack(key)) for key in data_keys])


def get_global_world_size():
    if is_xla():
        return xm.xrt_world_size()
    else:
        return torch.distributed.get_world_size()


def get_global_rank():
    if is_xla():
        return xm.get_ordinal()
    else:
        return torch.distributed.get_rank()


def initialize_distributed_groups(model_parallel_size, use_xla=False):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    NOTE: modified from megatron.mpu.initialize to support XLA/TPUs.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """

    # Get world size and rank. Ensure some consistencies.
    world_size = get_global_world_size()
    model_parallel_size = min(model_parallel_size, world_size)
    assert world_size % model_parallel_size == 0
    rank = get_global_rank()

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    grouped_ranks = []
    for i in range(model_parallel_size):
        ranks = list(range(i, world_size, model_parallel_size))
        grouped_ranks.append(ranks)
    _DATA_PARALLEL_GROUP = new_groups(grouped_ranks)

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    grouped_ranks = []
    for i in range(world_size // model_parallel_size):
        ranks = list(
            range(i * model_parallel_size, (i + 1) * model_parallel_size)
        )
        grouped_ranks.append(ranks)
    _MODEL_PARALLEL_GROUP = new_groups(grouped_ranks)

    if model_parallel_size > 1:
        logger.info(
            'initialized model parallel with size {}'.format(model_parallel_size)
        )


def get_model_parallel_group():
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def _find_my_group_index(grouped_ranks):
    my_rank = get_global_rank()
    my_group = None
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError


def _find_my_group(grouped_ranks):
    index = _find_my_group_index(grouped_ranks)
    return grouped_ranks[index]


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return get_world_size(get_model_parallel_group())


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return get_rank(get_model_parallel_group())


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_world_size(get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_rank(get_data_parallel_group())


@contextlib.contextmanager
def fork_rng_for_model_parallel():
    # use the data parallel RNG to generate the starting seed for the fork
    start_seed = torch.randint(1000000, (1,)).item()
    seed = start_seed + get_model_parallel_rank()
    with utils.set_torch_seed(seed):
        yield
