import importlib
import os
from typing import Optional, Dict
from abc import ABC, abstractmethod


class S2TTransform(ABC):
    @classmethod
    @abstractmethod
    def from_config_dict(cls, config: Optional[Dict] = None):
        pass


TRANSFORM_REGISTRY = {}
TRANSFORM_CLASS_NAMES = set()


def register_transform(name):
    def register_transform_cls(cls):
        if name in TRANSFORM_REGISTRY:
            raise ValueError(f'Cannot register duplicate transform ({name})')
        if not issubclass(cls, S2TTransform):
            raise ValueError(f'Transform ({name}: {cls.__name__}) must extend S2TTransform')
        if cls.__name__ in TRANSFORM_CLASS_NAMES:
            raise ValueError(f'Cannot register transform with duplicate class name ({cls.__name__})')
        TRANSFORM_REGISTRY[name] = cls
        TRANSFORM_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_transform_cls


def get_transform(name):
    return TRANSFORM_REGISTRY[name]


transforms_dir = os.path.dirname(__file__)
for file in os.listdir(transforms_dir):
    path = os.path.join(transforms_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        name = file[:file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('fairseq.data.audio.transforms.' + name)


class CompositeTransform(S2TTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        _transforms = _config.get('transforms')
        if _transforms is None:
            return None
        transforms = [get_transform(_t).from_config_dict(_config.get(_t))
                      for _t in _transforms]
        return CompositeTransform(transforms)

    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = [self.__class__.__name__ + '('] + \
                        [f"    {t.__repr__()}" for t in self.transforms] + [')']
        return '\n'.join(format_string)
