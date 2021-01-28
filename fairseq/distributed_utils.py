# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import os
import pickle
import random
import socket
import struct
import subprocess
import warnings
from argparse import Namespace
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist
from fairseq import utils
from fairseq.dataclass.configs import DistributedTrainingConfig, FairseqConfig
from omegaconf import open_dict

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


# Flag to indicate if we're using Megatron
# NOTE: this is a temporary hack until we move away from Megatron's model parallel init
_USE_MEGATRON = False

# Whether to use XLA ops (e.g., on TPUs) instead of CUDA ops.
_USE_XLA = False


logger = logging.getLogger(__name__)


def is_master(cfg: DistributedTrainingConfig):
    return cfg.distributed_rank == 0


def infer_init_method(cfg: DistributedTrainingConfig, force_distributed=False):
    if cfg.distributed_init_method is not None or cfg.tpu:
        return

    if cfg.pipeline_model_parallel:
        balance_exists = (
            cfg.pipeline_balance is not None
            or cfg.pipeline_encoder_balance is not None
            or cfg.pipeline_decoder_balance is not None
        )
        devices_exist = (
            cfg.pipeline_devices is not None
            or cfg.pipeline_encoder_devices is not None
            or cfg.pipeline_decoder_devices is not None
        )
        if not balance_exists:
            raise ValueError(
                "--pipeline-balance is currently required for pipeline model parallelism"
            )
        if not devices_exist:
            raise ValueError(
                "--pipeline-devices is currently required for pipeline model parallelism"
            )

        cfg.pipeline_balance = utils.eval_str_list(cfg.pipeline_balance, type=int)
        if cfg.pipeline_devices is not None:
            cfg.pipeline_devices = utils.eval_str_list(cfg.pipeline_devices, type=int)
            num_pipeline_devices = len(set(cfg.pipeline_devices))
        else:
            cfg.pipeline_encoder_devices = utils.eval_str_list(
                cfg.pipeline_encoder_devices, type=int
            )
            cfg.pipeline_decoder_devices = utils.eval_str_list(
                cfg.pipeline_decoder_devices, type=int
            )
            num_pipeline_devices = len(
                set(cfg.pipeline_encoder_devices + cfg.pipeline_decoder_devices)
            )
        gpus_per_node = torch.cuda.device_count()
        assert (
            gpus_per_node >= num_pipeline_devices
            and gpus_per_node % num_pipeline_devices == 0
        ), (
            "the number of unique device IDs in --pipeline-devices must evenly divide "
            "the number of GPUs per node (multi-node pipelining is not yet supported)"
        )
        num_pipelines_per_node = gpus_per_node // num_pipeline_devices

    # support torch.distributed.launch
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        cfg.distributed_init_method = "env://"
        cfg.distributed_world_size = int(os.environ["WORLD_SIZE"])
        cfg.distributed_rank = int(os.environ["RANK"])
        # processes are created by torch.distributed.launch
        cfg.distributed_no_spawn = True

    # we can determine the init method automatically for Slurm
    elif cfg.distributed_port > 0:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                cfg.distributed_init_method = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=cfg.distributed_port,
                )
                nnodes = int(os.environ.get("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    nnodes = int(os.environ.get("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    gpus_per_node = torch.cuda.device_count()
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    cfg.distributed_rank = node_id * gpus_per_node
                    cfg.distributed_world_size = nnodes * gpus_per_node
                elif cfg.pipeline_model_parallel:
                    assert ntasks_per_node == num_pipelines_per_node, (
                        "SLURM --ntasks-per-node must match number of pipelines per "
                        "node (={})".format(num_pipelines_per_node)
                    )
                    cfg.distributed_no_spawn = True
                    # For 4-way MP on nodes with 8 GPUs, ranks will be [0, 1] on
                    # the first node, [1, 2] on the second node, etc. This
                    # matches torch.distributed.launch.
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    local_id = int(os.environ.get("SLURM_LOCALID"))
                    cfg.distributed_rank = node_id * num_pipelines_per_node + local_id
                    # In the above example, device_id will always be in [0, 1],
                    # which also matches torch.distributed.launch.
                    cfg.device_id = local_id
                    # We also want to set distributed_world_size to be the total
                    # number of pipelines across all nodes.
                    cfg.distributed_world_size = nnodes * num_pipelines_per_node
                else:
                    assert ntasks_per_node == cfg.distributed_world_size // nnodes
                    cfg.distributed_no_spawn = True
                    cfg.distributed_rank = int(os.environ.get("SLURM_PROCID"))
                    cfg.device_id = int(os.environ.get("SLURM_LOCALID"))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass

    elif cfg.distributed_world_size > 1 or force_distributed:
        # fallback for single node with multiple GPUs
        assert (
            cfg.distributed_world_size <= torch.cuda.device_count()
        ), f"world size is {cfg.distributed_world_size} but have {torch.cuda.device_count()} available devices"
        port = random.randint(10000, 20000)
        cfg.distributed_init_method = "tcp://localhost:{port}".format(port=port)

    if cfg.pipeline_model_parallel:
        if not cfg.distributed_no_spawn:
            # When distributed_no_spawn is False, we expect distributed_rank and
            # distributed_world_size to be based on the total number of GPUs, so
            # we need to correct them to be based on the number of pipelines.
            assert cfg.distributed_world_size % num_pipeline_devices == 0
            cfg.distributed_world_size = (
                cfg.distributed_world_size // num_pipeline_devices
            )
            # In the case of 4-way MP on nodes with 8 GPUs, we want
            # distributed_rank to be the starting GPU index for each pipeline
            # i.e., 0, 2, ...
            assert cfg.distributed_rank % gpus_per_node == 0
            assert cfg.distributed_rank % num_pipeline_devices == 0

            with open_dict(cfg):
                cfg.distributed_rank = cfg.distributed_rank // num_pipeline_devices
                # launch one process per pipeline
                cfg.distributed_num_procs = num_pipelines_per_node

        # if we have 4-way MP on a node with 8 GPUs, we want device_ids to be 0
        # and 4, indicating the starting device IDs for each pipeline
        cfg.device_id *= num_pipeline_devices

        if cfg.device_id > 0:
            # if there's multiple pipelines on a node (e.g., 4-way MP on an 8
            # GPU node), we need to adjust pipeline_devices accordingly
            logger.debug(
                "setting CUDA device={} on rank {}".format(
                    cfg.device_id, cfg.distributed_rank
                )
            )
            torch.cuda.set_device(cfg.device_id)
            with open_dict(cfg):
                cfg.pipeline_devices = [cfg.device_id + d for d in cfg.pipeline_devices]
            logger.info(
                "setting pipeline_devices={} on rank {}".format(
                    cfg.pipeline_devices, cfg.distributed_rank
                )
            )
    elif not cfg.distributed_no_spawn:
        with open_dict(cfg):
            cfg.distributed_num_procs = min(
                torch.cuda.device_count(), cfg.distributed_world_size
            )


def distributed_init(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        from fairseq.dataclass.utils import convert_namespace_to_omegaconf

        cfg = convert_namespace_to_omegaconf(cfg)

    if not cfg.common.tpu:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            warnings.warn(
                "Distributed is already initialized, cannot initialize twice!"
            )
        else:
            logger.info(
                "distributed init (rank {}): {}".format(
                    cfg.distributed_training.distributed_rank,
                    cfg.distributed_training.distributed_init_method,
                )
            )
            dist.init_process_group(
                backend=cfg.distributed_training.distributed_backend,
                init_method=cfg.distributed_training.distributed_init_method,
                world_size=cfg.distributed_training.distributed_world_size,
                rank=cfg.distributed_training.distributed_rank,
            )
            logger.info(
                "initialized host {} as rank {}".format(
                    socket.gethostname(),
                    cfg.distributed_training.distributed_rank,
                )
            )

            # perform a dummy all-reduce to initialize the NCCL communicator
            if torch.cuda.is_available():
                dist.all_reduce(torch.zeros(1).cuda())

        cfg.distributed_training.distributed_rank = torch.distributed.get_rank()
    else:
        assert xm.xrt_world_size() == cfg.distributed_training.distributed_world_size
        global _USE_XLA
        _USE_XLA = True
        cfg.distributed_training.device_id = xm.get_local_ordinal()
        cfg.distributed_training.distributed_rank = xm.get_ordinal()
        xm.rendezvous("distributed_init")  # wait for all workers
        xm.mark_step()

    if is_master(cfg.distributed_training):
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    if cfg.common.model_parallel_size > 1:
        try:
            from fairseq.model_parallel.megatron.mpu import (
                initialize_model_parallel,
                model_parallel_cuda_manual_seed,
            )
        except ImportError:
            raise ImportError(
                "\n\nPlease install the megatron submodule:"
                "\n\n  git submodule update --init "
                "fairseq/model_parallel/megatron"
            )
        global _USE_MEGATRON
        _USE_MEGATRON = True
        initialize_model_parallel(cfg.common.model_parallel_size)
        model_parallel_cuda_manual_seed(cfg.common.seed)
        model_part_number = get_model_parallel_rank()
        cfg.checkpoint.checkpoint_suffix += "-model_part-{0}".format(model_part_number)

    return cfg.distributed_training.distributed_rank


def distributed_main(i, main, cfg: FairseqConfig, kwargs):
    cfg.distributed_training.device_id = i
    if torch.cuda.is_available() and not cfg.common.cpu and not cfg.common.tpu:
        torch.cuda.set_device(cfg.distributed_training.device_id)
    if cfg.distributed_training.distributed_rank is None:  # torch.multiprocessing.spawn
        cfg.distributed_training.distributed_rank = kwargs.pop("start_rank", 0) + i

    cfg.distributed_training.distributed_rank = distributed_init(cfg)

    after_distributed_init_fn = kwargs.pop("after_distributed_init_fn", None)
    if after_distributed_init_fn:
        cfg = after_distributed_init_fn(cfg)

    main(cfg, **kwargs)


def call_main(cfg: FairseqConfig, main, **kwargs):
    if cfg.distributed_training.distributed_init_method is None:
        infer_init_method(cfg.distributed_training)

    if cfg.distributed_training.distributed_init_method is not None:
        # distributed training
        if not cfg.distributed_training.distributed_no_spawn:
            start_rank = cfg.distributed_training.distributed_rank
            cfg.distributed_training.distributed_rank = None  # assign automatically
            kwargs["start_rank"] = start_rank
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(main, cfg, kwargs),
                nprocs=min(
                    torch.cuda.device_count(),
                    cfg.distributed_training.distributed_world_size,
                ),
                join=True,
            )
        else:
            distributed_main(cfg.distributed_training.device_id, main, cfg, kwargs)
    elif cfg.common.tpu and cfg.distributed_training.distributed_world_size > 1:
        import torch_xla.distributed.xla_multiprocessing as xmp

        torch.multiprocessing.set_sharing_strategy("file_system")
        xmp.spawn(
            fn=distributed_main,
            args=(main, cfg, kwargs),
            nprocs=8,  # use all 8 TPU cores
        )
    else:
        # single GPU main
        main(cfg, **kwargs)


def use_xla():
    global _USE_XLA
    return _USE_XLA


def new_groups(grouped_ranks: List[List[int]]):
    if use_xla():
        return ("tpu", grouped_ranks)
    else:
        groups = [dist.new_group(g) for g in grouped_ranks]
        my_group_idx = _find_my_group_index(grouped_ranks)
        return groups[my_group_idx]


def _find_my_group_index(grouped_ranks):
    my_rank = get_global_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError


def _find_my_group(grouped_ranks):
    index = _find_my_group_index(grouped_ranks)
    return grouped_ranks[index]


def get_rank(group):
    if use_xla():
        assert group[0] == "tpu"
        my_group = _find_my_group(group[1])
        return my_group.index(get_global_rank())
    else:
        return dist.get_rank(group=group)


def get_world_size(group):
    if use_xla():
        assert group[0] == "tpu"
        my_group = _find_my_group(group[1])
        return len(my_group)
    elif torch.distributed.is_initialized():
        return dist.get_world_size(group=group)
    else:
        return 1


def get_global_group():
    if use_xla():
        return new_groups([list(range(get_global_world_size()))])
    elif torch.distributed.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            # ideally we could use torch.distributed.group.WORLD, but it seems
            # to cause random NCCL hangs in some cases
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None


def get_global_rank():
    if use_xla():
        return xm.get_ordinal()
    elif torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_global_world_size():
    if use_xla():
        return xm.xrt_world_size()
    elif torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    global _USE_MEGATRON
    if _USE_MEGATRON:
        from fairseq.model_parallel.megatron import mpu

        return mpu.get_data_parallel_group()
    else:
        return get_global_group()


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_rank(get_data_parallel_group())


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_world_size(get_data_parallel_group())


def get_model_parallel_group():
    global _USE_MEGATRON
    if _USE_MEGATRON:
        from fairseq.model_parallel.megatron import mpu

        return mpu.get_model_parallel_group()
    else:
        return None


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return get_rank(get_model_parallel_group())


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return get_world_size(get_model_parallel_group())


def all_reduce(tensor, group, op="sum"):
    if use_xla():
        assert isinstance(group, tuple) and group[0] == "tpu"
        tensor = [tensor]  # wrap in a list to make xm.all_reduce in-place
        return xm.all_reduce(op, tensor, groups=group[1])[0]
    else:
        if op == "sum":
            op = dist.ReduceOp.SUM
        elif op == "max":
            op = dist.ReduceOp.MAX
        else:
            raise NotImplementedError
        dist.all_reduce(tensor, op=op, group=group)
        return tensor


def broadcast(tensor, src, group):
    if use_xla():
        # XLA doesn't support broadcast, hack it with all_reduce
        if get_rank(group) != src:
            tensor.zero_()
        all_reduce(tensor, group)
    else:
        dist.broadcast(tensor, src=src, group=group)


def all_to_all(tensor, group):
    """Perform an all-to-all operation on a 1D Tensor."""
    assert tensor.dim() == 1
    split_count = get_world_size(group=group)
    assert tensor.numel() % split_count == 0
    if use_xla():
        assert isinstance(group, tuple) and group[0] == "tpu"
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
    if use_xla():
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
            tensor if i == rank else torch.empty_like(tensor) for i in range(world_size)
        ]
        dist.all_gather(tensor_list, tensor, group=group)
        if return_tensor:
            return torch.stack(tensor_list, dim=0)
        else:
            return tensor_list


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    if group is None:
        group = get_global_group()
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)

    buffer_size = max_size * world_size
    if (
        not hasattr(all_gather_list, "_buffer")
        or all_gather_list._buffer.numel() < buffer_size
    ):
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
        raise ValueError(
            "encoded data size ({}) exceeds max_size ({})".format(size, max_size)
        )

    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = torch.ByteTensor(list(header + enc))
    start = rank * max_size
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            (enc_size,) = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(
                    pickle.loads(
                        bytes(out_buffer[header_size : header_size + enc_size].tolist())
                    )
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all "
            "workers to enter the function together, so this error usually indicates "
            "that the workers have fallen out of sync somehow. Workers can fall out of "
            "sync if one of them runs out of memory, or if there are other conditions "
            "in your training script that can cause one worker to finish an epoch "
            "while other workers are still iterating over their portions of the data. "
            "Try rerunning with --ddp-backend=legacy_ddp and see if that helps."
        )


def all_reduce_dict(data: Mapping[str, Any], device, group) -> Dict[str, Any]:
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
        buf = torch.cat([t.view(-1) for t in data.values()]).to(device=device)
        all_reduce(buf, group=group)
        split_buf = torch.split(buf, [t.numel() for t in data.values()])
        reduced_data = [t.view_as(orig) for t, orig in zip(split_buf, data.values())]
        return OrderedDict(zip(data.keys(), reduced_data))

    cpu_data = _all_reduce_dict(cpu_data)
    device_data = _all_reduce_dict(device_data)

    def get_from_stack(key):
        if key in cpu_data:
            return cpu_data[key]
        elif key in device_data:
            return device_data[key]
        raise KeyError

    return OrderedDict([(key, get_from_stack(key)) for key in data_keys])


def broadcast_tensors(
    tensors: Optional[List[torch.Tensor]],
    src_rank: int,
    group: object,
    dist_device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Broadcasts a list of tensors without other (non-src) ranks needing to know
    the dtypes/shapes of the tensors.
    """
    if dist_device is None:
        if torch.distributed.get_backend(group) == "nccl":
            dist_device = torch.device("cuda")
        else:
            dist_device = torch.device("cpu")

    # share metadata first to simplify transfer
    is_src_rank = (get_rank(group) == src_rank)
    if is_src_rank:
        metadata = [
            {"size": t.size(), "dtype": t.dtype, "device": t.device} for t in tensors
        ]
        metadata = _broadcast_object_slow(metadata, src_rank, group, dist_device)
    else:
        metadata = _broadcast_object_slow(None, src_rank, group, dist_device)

    out_tensors = []
    for i, meta in enumerate(metadata):
        if is_src_rank:
            tensor = tensors[i]
            broadcast(tensors[i].to(dist_device), src=src_rank, group=group)
        else:
            tensor = torch.zeros(
                [meta["size"].numel()], dtype=meta["dtype"], device=dist_device
            )
            broadcast(tensor, src=src_rank, group=group)
        tensor = tensor.view(meta["size"]).to(meta["device"])
        out_tensors.append(tensor)
    return out_tensors


def broadcast_object(
    obj: Any,
    src_rank: int,
    group: object,
    dist_device: Optional[torch.device] = None,
) -> Any:
    """Broadcast an arbitrary Python object to other workers."""
    if dist_device is None:
        if torch.distributed.get_backend(group) == "nccl":
            dist_device = torch.device("cuda")
        else:
            dist_device = torch.device("cpu")

    if get_rank(group) == src_rank:
        # split the tensors from the non-tensors so we can broadcast them
        # directly, avoiding unnecessary serialization/deserialization
        tensors = []
        obj = _split_tensors_from_obj(obj, tensors)
        obj = _broadcast_object_slow(obj, src_rank, group, dist_device)
        tensors = broadcast_tensors(tensors, src_rank, group, dist_device)
    else:
        obj = _broadcast_object_slow(None, src_rank, group, dist_device)
        tensors = broadcast_tensors(None, src_rank, group, dist_device)
    return _put_tensors_in_obj(obj, tensors)


def _broadcast_object_slow(
    obj: Any, src_rank: int, group: object, dist_device: torch.device,
) -> Any:
    if get_rank(group) == src_rank:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer = torch.ByteTensor(buffer.getbuffer()).to(dist_device)
        length = torch.LongTensor([len(buffer)]).to(dist_device)
        broadcast(length, src=src_rank, group=group)
        broadcast(buffer, src=src_rank, group=group)
    else:
        # Fetch from the source
        length = torch.LongTensor([0]).to(dist_device)
        broadcast(length, src=src_rank, group=group)
        buffer = torch.ByteTensor(int(length.item())).to(dist_device)
        broadcast(buffer, src=src_rank, group=group)
        buffer = io.BytesIO(buffer.cpu().numpy())
        obj = torch.load(buffer, map_location="cpu")
    return obj


@dataclass(frozen=True)
class _TensorPlaceholder:
    index: int


def _split_tensors_from_obj(obj: Any, tensors: List[torch.Tensor]) -> Any:
    if torch.is_tensor(obj):
        placeholder = _TensorPlaceholder(index=len(tensors))
        tensors.append(obj)
        return placeholder
    elif isinstance(obj, dict):
        return {k: _split_tensors_from_obj(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_split_tensors_from_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_split_tensors_from_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_split_tensors_from_obj(v, tensors) for v in obj}
    else:
        return obj


def _put_tensors_in_obj(obj: Any, tensors: List[torch.Tensor]) -> Any:
    if isinstance(obj, _TensorPlaceholder):
        return tensors[obj.index]
    elif isinstance(obj, dict):
        return {k: _put_tensors_in_obj(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_put_tensors_in_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_put_tensors_in_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_put_tensors_in_obj(v, tensors) for v in obj}
    else:
        return obj
