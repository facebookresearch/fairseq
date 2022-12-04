import importlib
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional


class AudioFeatureTransform(ABC):
    @classmethod
    @abstractmethod
    def from_config_dict(cls, config: Optional[Dict] = None):
        pass


AUDIO_FEATURE_TRANSFORM_REGISTRY = {}
AUDIO_FEATURE_TRANSFORM_CLASS_NAMES = set()


def register_audio_feature_transform(name):
    def register_audio_feature_transform_cls(cls):
        if name in AUDIO_FEATURE_TRANSFORM_REGISTRY:
            raise ValueError(f"Cannot register duplicate transform ({name})")
        if not issubclass(cls, AudioFeatureTransform):
            raise ValueError(
                f"Transform ({name}: {cls.__name__}) must extend "
                "AudioFeatureTransform"
            )
        if cls.__name__ in AUDIO_FEATURE_TRANSFORM_CLASS_NAMES:
            raise ValueError(
                f"Cannot register audio feature transform with duplicate "
                f"class name ({cls.__name__})"
            )
        AUDIO_FEATURE_TRANSFORM_REGISTRY[name] = cls
        AUDIO_FEATURE_TRANSFORM_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_audio_feature_transform_cls


def get_audio_feature_transform(name):
    return AUDIO_FEATURE_TRANSFORM_REGISTRY[name]


transforms_dir = os.path.dirname(__file__)
for file in os.listdir(transforms_dir):
    path = os.path.join(transforms_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("fairseq.data.audio.feature_transforms." + name)


class CompositeAudioFeatureTransform(AudioFeatureTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        _transforms = _config.get("transforms")
        if _transforms is None:
            return None
        transforms = [
            get_audio_feature_transform(_t).from_config_dict(_config.get(_t))
            for _t in _transforms
        ]
        return CompositeAudioFeatureTransform(transforms)

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
