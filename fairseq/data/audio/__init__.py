from abc import ABC, abstractmethod
from typing import Dict, Optional
import importlib
import os
import numpy as np


class AudioTransform(ABC):
    @classmethod
    @abstractmethod
    def from_config_dict(cls, config: Optional[Dict] = None):
        pass


class CompositeAudioTransform(AudioTransform):
    def _from_config_dict(
        cls,
        transform_type,
        get_audio_transform,
        composite_cls,
        config=None,
        return_empty=False,
    ):
        _config = {} if config is None else config
        _transforms = _config.get(f"{transform_type}_transforms")

        if _transforms is None:
            if return_empty:
                _transforms = []
            else:
                return None

        transforms = [
            get_audio_transform(_t).from_config_dict(_config.get(_t))
            for _t in _transforms
        ]
        return composite_cls(transforms)

    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = (
            [self.__class__.__name__ + "("]
            + [f"    {t.__repr__()}" for t in self.transforms]
            + [")"]
        )
        return "\n".join(format_string)


def register_audio_transform(name, cls_type, registry, class_names):
    def register_audio_transform_cls(cls):
        if name in registry:
            raise ValueError(f"Cannot register duplicate transform ({name})")
        if not issubclass(cls, cls_type):
            raise ValueError(
                f"Transform ({name}: {cls.__name__}) must extend "
                f"{cls_type.__name__}"
            )
        if cls.__name__ in class_names:
            raise ValueError(
                f"Cannot register audio transform with duplicate "
                f"class name ({cls.__name__})"
            )
        registry[name] = cls
        class_names.add(cls.__name__)
        return cls

    return register_audio_transform_cls


def import_transforms(transforms_dir, transform_type):
    for file in os.listdir(transforms_dir):
        path = os.path.join(transforms_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(
                f"fairseq.data.audio.{transform_type}_transforms." + name
            )


# Utility fn for uniform numbers in transforms
def rand_uniform(a, b):
    return np.random.uniform() * (b - a) + a
