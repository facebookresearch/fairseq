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

    def __call__(self, x, sample_rate):
        for t in self.transforms:
            x, sample_rate = t(x, sample_rate)
        return x, sample_rate
