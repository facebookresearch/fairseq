import importlib
import os
from fairseq.data.audio import AudioTransform, CompositeAudioTransform, register_audio_transform


class AudioFeatureTransform(AudioTransform):
    pass


AUDIO_FEATURE_TRANSFORM_REGISTRY = {}
AUDIO_FEATURE_TRANSFORM_CLASS_NAMES = set()


def get_audio_feature_transform(name):
    return AUDIO_FEATURE_TRANSFORM_REGISTRY[name]


def register_audio_feature_transform(name):
    return register_audio_transform(
        name,
        AudioFeatureTransform,
        AUDIO_FEATURE_TRANSFORM_REGISTRY,
        AUDIO_FEATURE_TRANSFORM_CLASS_NAMES,
    )


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


class CompositeAudioFeatureTransform(CompositeAudioTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        return super()._from_config_dict(
            cls,
            "feature",
            get_audio_feature_transform,
            CompositeAudioFeatureTransform,
            config,
        )
