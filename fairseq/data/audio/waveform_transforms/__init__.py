import os
from fairseq.data.audio import (
    AudioTransform,
    CompositeAudioTransform,
    import_transforms,
    register_audio_transform,
)


class AudioWaveformTransform(AudioTransform):
    pass


AUDIO_WAVEFORM_TRANSFORM_REGISTRY = {}
AUDIO_WAVEFORM_TRANSFORM_CLASS_NAMES = set()


def get_audio_waveform_transform(name):
    return AUDIO_WAVEFORM_TRANSFORM_REGISTRY[name]


def register_audio_waveform_transform(name):
    return register_audio_transform(
        name,
        AudioWaveformTransform,
        AUDIO_WAVEFORM_TRANSFORM_REGISTRY,
        AUDIO_WAVEFORM_TRANSFORM_CLASS_NAMES,
    )


import_transforms(os.path.dirname(__file__), "waveform")


class CompositeAudioWaveformTransform(CompositeAudioTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        return super()._from_config_dict(
            cls,
            "waveform",
            get_audio_waveform_transform,
            CompositeAudioWaveformTransform,
            config,
        )

    def __call__(self, x, always_2d=False, use_sample_rate=None):
        for t in self.transforms:
            x = t(x, always_2d, use_sample_rate)
        return x
