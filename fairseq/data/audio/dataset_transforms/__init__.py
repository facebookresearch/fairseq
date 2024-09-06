import os
from fairseq.data.audio import (
    AudioTransform,
    CompositeAudioTransform,
    import_transforms,
    register_audio_transform,
)


class AudioDatasetTransform(AudioTransform):
    pass


AUDIO_DATASET_TRANSFORM_REGISTRY = {}
AUDIO_DATASET_TRANSFORM_CLASS_NAMES = set()


def get_audio_dataset_transform(name):
    return AUDIO_DATASET_TRANSFORM_REGISTRY[name]


def register_audio_dataset_transform(name):
    return register_audio_transform(
        name,
        AudioDatasetTransform,
        AUDIO_DATASET_TRANSFORM_REGISTRY,
        AUDIO_DATASET_TRANSFORM_CLASS_NAMES,
    )


import_transforms(os.path.dirname(__file__), "dataset")


class CompositeAudioDatasetTransform(CompositeAudioTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        return super()._from_config_dict(
            cls,
            "dataset",
            get_audio_dataset_transform,
            CompositeAudioDatasetTransform,
            config,
            return_empty=True,
        )

    def get_transform(self, cls):
        for t in self.transforms:
            if isinstance(t, cls):
                return t
        return None

    def has_transform(self, cls):
        return self.get_transform(cls) is not None
