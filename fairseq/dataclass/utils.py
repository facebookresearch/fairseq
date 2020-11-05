# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import os
from argparse import ArgumentError, ArgumentParser, Namespace
from dataclasses import _MISSING_TYPE, MISSING
from enum import Enum
import inspect
from typing import Any, Dict, List, Tuple, Type

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import FairseqConfig
from hydra.experimental import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict


def eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return []
        x = ast.literal_eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]


def gen_parser_from_dataclass(
    parser: ArgumentParser,
    dataclass_instance: FairseqDataclass,
    delete_default: bool = False,
) -> None:
    """convert a dataclass instance to tailing parser arguments"""
    import re

    def argparse_name(name: str):
        if name == "data":
            # normally data is positional args
            return name
        if name == "_name":
            # private member, skip
            return None
        return "--" + name.replace("_", "-")

    def interpret_dc_type(field_type):
        if isinstance(field_type, str):
            raise RuntimeError("field should be a type")
        typestring = str(field_type)
        if re.match(r"(typing.|^)Union\[(.*), NoneType\]$", typestring):
            return field_type.__args__[0]
        return field_type

    def get_kwargs_from_dc(
        dataclass_instance: FairseqDataclass, k: str
    ) -> Dict[str, Any]:
        """k: dataclass attributes"""

        kwargs = {}

        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)

        field_default = dataclass_instance._get_default(k)

        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None

        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)

        if isinstance(field_default, str) and field_default.startswith("${"):
            kwargs["default"] = field_default
        else:
            if field_default is MISSING:
                kwargs["required"] = True
            if field_choices is not None:
                kwargs["choices"] = field_choices
            if (isinstance(inter_type, type) and issubclass(inter_type, List)) or (
                "List" in str(inter_type)
            ):
                if "int" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, int)
                elif "float" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, float)
                elif "str" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError()
                if field_default is not MISSING:
                    kwargs["default"] = (
                        ",".join(map(str, field_default))
                        if field_default is not None
                        else None
                    )
            elif (
                isinstance(inter_type, type) and issubclass(inter_type, Enum)
            ) or "Enum" in str(inter_type):
                kwargs["type"] = str
                if field_default is not MISSING:
                    if isinstance(field_default, Enum):
                        kwargs["default"] = field_default.value
                    else:
                        kwargs["default"] = field_default
            elif inter_type is bool:
                kwargs["action"] = (
                    "store_false" if field_default is True else "store_true"
                )
                kwargs["default"] = field_default
            else:
                kwargs["type"] = inter_type
                if field_default is not MISSING:
                    kwargs["default"] = field_default

        kwargs["help"] = field_help
        if field_const is not None:
            kwargs["const"] = field_const
            kwargs["nargs"] = "?"

        return kwargs

    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        if field_name is None:
            continue
        kwargs = get_kwargs_from_dc(dataclass_instance, k)

        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)

        if "default" in kwargs:
            if isinstance(kwargs["default"], str) and kwargs["default"].startswith(
                "${"
            ):
                continue
            if delete_default:
                del kwargs["default"]
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass


def _set_legacy_defaults(args, cls):
    """Helper to set default arguments based on *add_args*."""
    if not hasattr(cls, "add_args"):
        return

    import argparse

    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, allow_abbrev=False
    )
    cls.add_args(parser)
    # copied from argparse.py:
    defaults = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(defaults, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(defaults, action.dest, action.default)
    for key, default_value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, default_value)


def _override_attr(
    sub_node: str, data_class: Type[FairseqDataclass], args: Namespace
) -> List[str]:
    overrides = []

    if not inspect.isclass(data_class) or not issubclass(data_class, FairseqDataclass):
        return overrides

    def get_default(f):
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    for k, v in data_class.__dataclass_fields__.items():
        if k.startswith("_"):
            # private member, skip
            continue

        val = get_default(v) if not hasattr(args, k) else getattr(args, k)

        if getattr(v.type, "__origin__", None) is List:
            # if type is int but val is float, then we will crash later - try to convert here
            t_args = v.type.__args__
            if len(t_args) == 1:
                val = list(map(t_args[0], val))

        if val is None:
            overrides.append("{}.{}=null".format(sub_node, k))
        elif val == "":
            overrides.append("{}.{}=''".format(sub_node, k))
        elif isinstance(val, str):
            overrides.append("{}.{}='{}'".format(sub_node, k, val))
        else:
            overrides.append("{}.{}={}".format(sub_node, k, val))
    return overrides


def migrate_registry(
    name, value, registry, args, overrides, deletes, use_name_as_val=False
):
    if value in registry:
        overrides.append("{}={}".format(name, value))
        overrides.append("{}._name={}".format(name, value))
        overrides.extend(_override_attr(name, registry[value], args))
    elif use_name_as_val and value is not None:
        overrides.append("{}={}".format(name, value))
    else:
        deletes.append(name)


def override_module_args(args: Namespace) -> Tuple[List[str], List[str]]:
    """use the field in args to overrides those in cfg"""
    overrides = []
    deletes = []

    for k in FairseqConfig.__dataclass_fields__.keys():
        overrides.extend(
            _override_attr(k, FairseqConfig.__dataclass_fields__[k].type, args)
        )

    if args is not None:
        if hasattr(args, "task"):
            from fairseq.tasks import TASK_DATACLASS_REGISTRY

            migrate_registry(
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
                migrate_registry(
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
                    overrides.extend(_override_attr("model", dc, args))
                    no_dc = False
        if no_dc:
            deletes.append("model")

    return overrides, deletes


def convert_namespace_to_omegaconf(args: Namespace) -> DictConfig:
    """Convert a flat argparse.Namespace to a structured DictConfig."""

    # Here we are using field values provided in args to override counterparts inside config object
    overrides, deletes = override_module_args(args)

    # configs will be in fairseq/config after installation
    config_path = os.path.join("..", "config")
    if not os.path.exists(config_path):
        # in case of "--editable" installs we need to go one dir up
        config_path = os.path.join("..", "..", "config")

    with initialize(config_path=config_path, strict=True):
        composed_cfg = compose("config", overrides=overrides, strict=False)
        for k in deletes:
            composed_cfg[k] = None

    cfg = OmegaConf.create(
        OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
    )

    # hack to be able to set Namespace in dict config. this should be removed when we update to newer
    # omegaconf version that supports object flags, or when we migrate all existing models
    from omegaconf import _utils

    old_primitive = _utils.is_primitive_type
    _utils.is_primitive_type = lambda _: True

    if cfg.task is None and getattr(args, "task", None):
        cfg.task = Namespace(**vars(args))
        from fairseq.tasks import TASK_REGISTRY

        _set_legacy_defaults(cfg.task, TASK_REGISTRY[args.task])
        cfg.task._name = args.task
    if cfg.model is None and getattr(args, "arch", None):
        cfg.model = Namespace(**vars(args))
        from fairseq.models import ARCH_MODEL_REGISTRY

        _set_legacy_defaults(cfg.model, ARCH_MODEL_REGISTRY[args.arch])
        cfg.model._name = args.arch
    if cfg.optimizer is None and getattr(args, "optimizer", None):
        cfg.optimizer = Namespace(**vars(args))
        from fairseq.optim import OPTIMIZER_REGISTRY

        _set_legacy_defaults(cfg.optimizer, OPTIMIZER_REGISTRY[args.optimizer])
        cfg.optimizer._name = args.optimizer
    if cfg.lr_scheduler is None and getattr(args, "lr_scheduler", None):
        cfg.lr_scheduler = Namespace(**vars(args))
        from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

        _set_legacy_defaults(cfg.lr_scheduler, LR_SCHEDULER_REGISTRY[args.lr_scheduler])
        cfg.lr_scheduler._name = args.lr_scheduler
    if cfg.criterion is None and getattr(args, "criterion", None):
        cfg.criterion = Namespace(**vars(args))
        from fairseq.criterions import CRITERION_REGISTRY

        _set_legacy_defaults(cfg.criterion, CRITERION_REGISTRY[args.criterion])
        cfg.criterion._name = args.criterion

    _utils.is_primitive_type = old_primitive
    OmegaConf.set_struct(cfg, True)
    return cfg


def populate_dataclass(
    args: Namespace, dataclass: FairseqDataclass
) -> FairseqDataclass:
    for k in dataclass.__dataclass_fields__.keys():
        if k.startswith("_"):
            # private member, skip
            continue
        if hasattr(args, k):
            setattr(dataclass, k, getattr(args, k))

        return dataclass


def overwrite_args_by_name(cfg: DictConfig, overrides: Dict[str, any]):
    # this will be deprecated when we get rid of argparse and model_overrides logic

    with open_dict(cfg):
        for k in cfg.keys():
            if isinstance(cfg[k], DictConfig):
                overwrite_args_by_name(cfg[k], overrides)
            elif k in overrides:
                cfg[k] = overrides[k]


def merge_with_parent(dc: FairseqDataclass, cfg: DictConfig):
    dc_instance = DictConfig(dc)
    dc_instance.__dict__["_parent"] = cfg.__dict__["_parent"]
    cfg = OmegaConf.merge(dc_instance, cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg
