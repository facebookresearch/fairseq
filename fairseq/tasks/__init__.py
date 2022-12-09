# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import argparse
import importlib
import os

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import merge_with_parent
from hydra.core.config_store import ConfigStore

from .fairseq_task import FairseqTask, LegacyFairseqTask  # noqa


# register dataclass
TASK_DATACLASS_REGISTRY = {}
TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


def setup_task(cfg: FairseqDataclass, **kwargs):
    task = None
    task_name = getattr(cfg, "task", None)

    if isinstance(task_name, str):
        # legacy tasks
        task = TASK_REGISTRY[task_name]
        if task_name in TASK_DATACLASS_REGISTRY:
            dc = TASK_DATACLASS_REGISTRY[task_name]
            cfg = dc.from_namespace(cfg)
    else:
        task_name = getattr(cfg, "_name", None)

        if task_name and task_name in TASK_DATACLASS_REGISTRY:
            remove_missing = "from_checkpoint" in kwargs and kwargs["from_checkpoint"]
            dc = TASK_DATACLASS_REGISTRY[task_name]
            cfg = merge_with_parent(dc(), cfg, remove_missing=remove_missing)
            task = TASK_REGISTRY[task_name]

    assert (
        task is not None
    ), f"Could not infer task type from {cfg}. Available argparse tasks: {TASK_REGISTRY.keys()}. Available hydra tasks: {TASK_DATACLASS_REGISTRY.keys()}"

    return task.setup_task(cfg, **kwargs)


def register_task(name, dataclass=None):
    """
    New tasks can be added to fairseq with the
    :func:`~fairseq.tasks.register_task` function decorator.

    For example::

        @register_task('classification')
        class ClassificationTask(FairseqTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~fairseq.tasks.FairseqTask`
        interface.

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            return TASK_REGISTRY[name]

        if not issubclass(cls, FairseqTask):
            raise ValueError(
                "Task ({}: {}) must extend FairseqTask".format(name, cls.__name__)
            )
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)

        if dataclass is not None and not issubclass(dataclass, FairseqDataclass):
            raise ValueError(
                "Dataclass {} must extend FairseqDataclass".format(dataclass)
            )

        cls.__dataclass = dataclass
        if dataclass is not None:
            TASK_DATACLASS_REGISTRY[name] = dataclass

            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="task", node=node, provider="fairseq")

        return cls

    return register_task_cls


def get_task(name):
    return TASK_REGISTRY[name]


def import_tasks(tasks_dir, namespace):
    for file in os.listdir(tasks_dir):
        path = os.path.join(tasks_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            task_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + task_name)

            # expose `task_parser` for sphinx
            if task_name in TASK_REGISTRY:
                parser = argparse.ArgumentParser(add_help=False)
                group_task = parser.add_argument_group("Task name")
                # fmt: off
                group_task.add_argument('--task', metavar=task_name,
                                        help='Enable this task with: ``--task=' + task_name + '``')
                # fmt: on
                group_args = parser.add_argument_group(
                    "Additional command-line arguments"
                )
                TASK_REGISTRY[task_name].add_args(group_args)
                globals()[task_name + "_parser"] = parser


# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
import_tasks(tasks_dir, "fairseq.tasks")
