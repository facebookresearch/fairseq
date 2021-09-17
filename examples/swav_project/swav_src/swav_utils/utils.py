from fairseq.dataclass.configs import FairseqConfig
import logging
import os
import sys
from argparse import Namespace

import torch
from fairseq import utils
from fairseq.dataclass import utils as dataclass_utils
from typing import Any, Callable, List, Optional, Tuple

from omegaconf import DictConfig

import pickle

import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# NOTE -------------- SWAV FUNCTIONS UTILS  ----------------------------------------------------
def maybe_improve_stability(x: torch.Tensor, world_size: int, improve_numerical_stability=True, affect_global=False):
    if improve_numerical_stability:
        if affect_global:
            _max = torch.max(x)
            if world_size > 1:
                dist.all_reduce(_max, op=dist.ReduceOp.MAX)
            x -= _max
            x = x.nan_to_num_()
        else:
            _max = x.max(-1, keepdim=True).values
            x -= _max
            x = x.nan_to_num_()
    return x


def check_num_stability(prefix: str, x: torch.Tensor):
    assert not torch.any(torch.isinf(x)), f'{prefix}: inf: {x} / {torch.isinf(x).int().sum()} / {x.numel()}'
    assert not torch.any(torch.isnan(x)), f'{prefix}: nan: {x}'


def dist_cat(world_size: int, tensor: torch.Tensor, dim: int = 0, same_size: bool = False):
    """
    For NCCL backend (default) all objects are moved to GPU anyway
        so make sure things to have enough memory for that
    * note: all_gather_object is not as efficient as it seems compared to dist.all_gather
        under the hood, it convert obj -> tensor.to(cuda) and resize each item to max_size of the group
        then use all_gather and than reduce the output size back to original
        and finally convert it back to objects
        --> This may cause huge CUDA OOM error as `tensor` is big enough, even if it is on CPU.
    """
    if world_size > 1:
        # NOTE: cannot use all_gather_object, must use all_gather
        if same_size:
            # preferred becausse it's faster
            t_list = [tensor.new(tensor.size()) for i in range(world_size)]
            dist.all_gather(t_list, tensor)
        else:
            # if GPU tensors, this will cause processes to access inter-gpu tensors
            #   e.g: proc 0 access gpu-7 tensor, and this will reflect in nvidia-smi
            device = tensor.device
            t_list = [None] * world_size
            dist.all_gather_object(t_list, tensor)
            t_list = [x.to(device=device) for x in t_list]
        t_cat = torch.cat(t_list, dim=dim)
        return t_cat
    else:
        return tensor


def dist_list(world_size: int, obj: Any, cat_fn: Optional[Callable] = None, same_size: bool = False):
    """
    For NCCL backend (default) all objects are moved to GPU anyway
        so make sure things to have enough memory for that
    """
    if world_size > 1:
        if same_size:
            assert isinstance(obj, torch.Tensor)
            o_list = [obj.new(obj.size()) for i in range(world_size)]
            dist.all_gather(o_list, obj)
        else:
            o_list = [None] * world_size
            dist.all_gather_object(o_list, obj)
        if cat_fn is not None:
            o_list = cat_fn(o_list)
        return o_list
    else:
        return obj


def object_to_tensor(obj):
    buffer = pickle.dumps(obj)
    byte_storage = torch.ByteStorage.from_buffer(buffer)
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def tensor_to_object(tensor, tensor_size):
    buf = tensor.numpy().tobytes()[:tensor_size]
    out = pickle.loads(buf)
    return out


def dist_batch_sizes(world_size: int, tensor: torch.Tensor, dim: int = 0):
    return utils.move_to_cpu(dist_list(
        world_size, torch.tensor([tensor.size(dim)]).long().cuda(), same_size=True)
    )


def dist_byte_sizes(world_size: int, tensor: torch.Tensor):
    return utils.move_to_cpu(dist_list(
        world_size, tensor.cuda(), same_size=True)
    )


@torch.no_grad()
def sinkhorn(
    Q: torch.Tensor, 
    nmb_iters: int, 
    world_size: int, 
    stability_epsilon: float = 0.0, 
    enforce_sum_ones: bool = False
):
    if world_size <= 1:
        return single_sinkhorn(Q, nmb_iters, enforce_sum_ones=enforce_sum_ones)
    try:
        """
        Issues with prototypes
        # new version from SWAV paper
        * even for float
            x.exp(-10) already give < 1e-5 values
            x.exp(-100) would give absolute zeros
        * feat_dim in CV swav loss is the reduction of dimensionality from 1024->128
        * need to check if q after exp() > 1
            because the maximum is 0, e^0 = 1, the rest should be < 1 until 0
        """
        # NOTE Q must be after q.exp()
        Q = Q.t()
        B = torch.tensor([Q.size(1)]).long().to(Q.device)
        dist.all_reduce(B)
        B = B.item()

        K = Q.shape[0]  # how many prototypes
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q + stability_epsilon
        for it in range(nmb_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)     # u in old version
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows + stability_epsilon   # look out for instability
            Q /= K
            # Q = shoot_infs(Q)
            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True) + stability_epsilon
            Q /= B
            check_num_stability(f'Iter {it} Q after sinkhorn', Q)
            # Q = shoot_infs(Q)
        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        Q = Q.t()
        # Q = shoot_infs(Q)
        check_num_stability('Q after sinkhorn', Q)
        if enforce_sum_ones:
            assert torch.all(torch.isclose(Q.sum(-1), Q.new(1).fill_(1), atol=1e-3)), f'invalid {Q.sum(-1)=}'
        return Q
    except Exception as e:
        logger.warning(
            "assertion error when process Q {}, wd_size: {}".format(
                Q.size(), world_size
            )
        )
        raise e


@torch.no_grad()
def single_sinkhorn(
    Q: torch.Tensor, 
    nmb_iters: int, 
    stability_epsilon: float = 0.0, 
    enforce_sum_ones: bool = False
):
    # NOTE Q must be after q.exp()
    Q = Q.t()
    # new version
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes
    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q + stability_epsilon
    for it in range(nmb_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)     # u in old version
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows + stability_epsilon   # look out for instability
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True) + stability_epsilon
        Q /= B
        check_num_stability(f'Iter {it} Q after sinkhorn', Q)
    Q *= B   # the colomns must sum to 1 so that Q is an assignment
    Q = Q.t()
    check_num_stability('Q after sinkhorn', Q)
    q_sum_ones = torch.isclose(Q.sum(-1), Q.new(1).fill_(1), atol=1e-3)
    assert not enforce_sum_ones or torch.all(q_sum_ones), f'invalid {Q.sum(-1)=}'
    return Q


def shoot_infs(inp_tensor: torch.Tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def infer_dist_params(key_word_size='world_size', key_rank='rank', **kwargs):
    """Search in kwargs and get world_size and rank and rank_reprs
    e.g:
    world_size, rank, rank_reprs = infer_dist_params(**kwargs)
    """
    world_size, rank = kwargs.get(key_word_size, 1), kwargs.get(key_rank, 0)
    rank_reprs = f'[{rank}/{world_size}]'
    return world_size, rank, rank_reprs


def iterative_broadcast_process(
    rank: int, world_size: int, tensors: List[torch.Tensor], 
    sizes: List[torch.Tensor], proc_fn: Callable, to_cpu=True, **kwargs
):
    assert len(tensors) == len(sizes), f'{len(tensors)=} != {len(sizes)=}'
    assert all(len(x) == world_size for x in sizes), f'{sizes}'
    rank_reprs = f'[{rank}/{world_size}]'

    gpu_tensors = utils.move_to_cuda(tensors)
    # iterative broad cast process
    outputs = None
    for r in range(world_size):
        is_sender = (r == rank)
        if is_sender:
            _tensors = [x.clone() for x in gpu_tensors]
        else:
            # _tensors = [x.new(*ss[r]) for i, (x, ss) in enumerate(zip(gpu_tensors, sizes))]
            _tensors = []
            for i, (x, ss) in enumerate(zip(gpu_tensors, sizes)):
                try:
                    assert len(ss) == world_size
                    _size = ss[r]
                    _t = x.new(*_size.tolist())
                    _tensors.append(_t)
                except Exception as e:
                    logger.warning(f'iter{rank_reprs}: {i}, {x.dtype} {x.size()}, _size={_size}')
                    raise e
        for x in _tensors:
            dist.broadcast(tensor=x, src=r)
        # logger.warning(f'{rank_reprs} Broadcast {r} is_sender={is_sender}')
        if to_cpu:
            rec_tensors = utils.move_to_cpu(_tensors)
        r_outputs = proc_fn(r, rec_tensors)
        # logger.warning(f'{rank_reprs} Finish Compute broadcast process rank={r}')

        if outputs is None:
            outputs = [[x] for x in r_outputs]
        else:
            for i, x in enumerate(r_outputs):
                outputs[i].append(x)
        torch.cuda.empty_cache()
    return outputs


# NOTE ---- MISCELLANEOUS ----------------
def custom_override_module_args(args: Namespace, config_class=FairseqConfig) -> Tuple[List[str], List[str]]:
    """use the field in args to overrides those in cfg"""
    overrides = []
    deletes = []

    for k in config_class.__dataclass_fields__.keys():
        overrides.extend(
            dataclass_utils._override_attr(k, config_class.__dataclass_fields__[k].type, args)
        )

    if args is not None:
        if hasattr(args, "task"):
            from fairseq.tasks import TASK_DATACLASS_REGISTRY

            dataclass_utils.migrate_registry(
                "task", args.task, TASK_DATACLASS_REGISTRY, args, overrides, deletes
            )
        else:
            deletes.append("task")

        # these options will be set to "None" if they have not yet been migrated
        # so we can populate them with the entire flat args
        CORE_REGISTRIES = {"criterion", "optimizer", "lr_scheduler"}

        from fairseq.registry import REGISTRIES

        for k, v in REGISTRIES.items():
            if hasattr(args, k):
                dataclass_utils.migrate_registry(
                    k,
                    getattr(args, k),
                    v["dataclass_registry"],
                    args,
                    overrides,
                    deletes,
                    use_name_as_val=k not in CORE_REGISTRIES,
                )
            else:
                deletes.append(k)

        no_dc = True
        if hasattr(args, "arch"):
            from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_MODEL_NAME_REGISTRY

            if args.arch in ARCH_MODEL_REGISTRY:
                m_cls = ARCH_MODEL_REGISTRY[args.arch]
                dc = getattr(m_cls, "__dataclass", None)
                if dc is not None:
                    m_name = ARCH_MODEL_NAME_REGISTRY[args.arch]
                    overrides.append("model={}".format(m_name))
                    overrides.append("model._name={}".format(args.arch))
                    # override model params with those exist in args
                    overrides.extend(dataclass_utils._override_attr("model", dc, args))
                    no_dc = False
        if no_dc:
            deletes.append("model")

    return overrides, deletes


def custom_convert_namespace_to_omegaconf(args: Namespace, override_module_args=None) -> DictConfig:
    # Here we are using field values provided in args to override counterparts inside config object
    if override_module_args is None:
        override_module_args = dataclass_utils.override_module_args
        # override_module_args = functools.partial(custom_override_module_args, config_class=GatherParaDataFairseqConfig)
    overrides, deletes = override_module_args(args)

    # configs will be in fairseq/config after installation
    config_path = os.path.join("..", "config")
    # config_path = os.path.join(".", "config")

    dataclass_utils.GlobalHydra.instance().clear()

    with dataclass_utils.initialize(config_path=config_path):
        try:
            composed_cfg = dataclass_utils.compose("config", overrides=overrides, strict=False)
        except Exception:
            logger.error("Error when composing. Overrides: " + str(overrides))
            raise

        for k in deletes:
            composed_cfg[k] = None

    cfg = dataclass_utils.OmegaConf.create(
        dataclass_utils.OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
    )

    # hack to be able to set Namespace in dict config. this should be removed when we update to newer
    # omegaconf version that supports object flags, or when we migrate all existing models
    from omegaconf import _utils

    old_primitive = _utils.is_primitive_type
    _utils.is_primitive_type = lambda _: True

    if cfg.task is None and getattr(args, "task", None):
        cfg.task = Namespace(**vars(args))
        from fairseq.tasks import TASK_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.task, TASK_REGISTRY[args.task])
        cfg.task._name = args.task
    if cfg.model is None and getattr(args, "arch", None):
        cfg.model = Namespace(**vars(args))
        from fairseq.models import ARCH_MODEL_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.model, ARCH_MODEL_REGISTRY[args.arch])
        cfg.model._name = args.arch
    if cfg.optimizer is None and getattr(args, "optimizer", None):
        cfg.optimizer = Namespace(**vars(args))
        from fairseq.optim import OPTIMIZER_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.optimizer, OPTIMIZER_REGISTRY[args.optimizer])
        cfg.optimizer._name = args.optimizer
    if cfg.lr_scheduler is None and getattr(args, "lr_scheduler", None):
        cfg.lr_scheduler = Namespace(**vars(args))
        from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.lr_scheduler, LR_SCHEDULER_REGISTRY[args.lr_scheduler])
        cfg.lr_scheduler._name = args.lr_scheduler
    if cfg.criterion is None and getattr(args, "criterion", None):
        cfg.criterion = Namespace(**vars(args))
        from fairseq.criterions import CRITERION_REGISTRY

        dataclass_utils._set_legacy_defaults(cfg.criterion, CRITERION_REGISTRY[args.criterion])
        cfg.criterion._name = args.criterion

    _utils.is_primitive_type = old_primitive
    dataclass_utils.OmegaConf.set_struct(cfg, True)
    return cfg



