import numpy as np
import torch

from fairseq.data.audio import rand_uniform
from fairseq.data.audio.dataset_transforms import (
    AudioDatasetTransform,
    register_audio_dataset_transform,
)
from fairseq.data.audio.waveform_transforms.noiseaugment import (
    NoiseAugmentTransform,
)

_DEFAULTS = {
    "rate": 0.25,
    "mixing_noise_rate": 0.1,
    "noise_path": "",
    "noise_snr_min": -5,
    "noise_snr_max": 5,
    "utterance_snr_min": -5,
    "utterance_snr_max": 5,
}


@register_audio_dataset_transform("noisyoverlapaugment")
class NoisyOverlapAugment(AudioDatasetTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return NoisyOverlapAugment(
            _config.get("rate", _DEFAULTS["rate"]),
            _config.get("mixing_noise_rate", _DEFAULTS["mixing_noise_rate"]),
            _config.get("noise_path", _DEFAULTS["noise_path"]),
            _config.get("noise_snr_min", _DEFAULTS["noise_snr_min"]),
            _config.get("noise_snr_max", _DEFAULTS["noise_snr_max"]),
            _config.get("utterance_snr_min", _DEFAULTS["utterance_snr_min"]),
            _config.get("utterance_snr_max", _DEFAULTS["utterance_snr_max"]),
        )

    def __init__(
        self,
        rate=_DEFAULTS["rate"],
        mixing_noise_rate=_DEFAULTS["mixing_noise_rate"],
        noise_path=_DEFAULTS["noise_path"],
        noise_snr_min=_DEFAULTS["noise_snr_min"],
        noise_snr_max=_DEFAULTS["noise_snr_max"],
        utterance_snr_min=_DEFAULTS["utterance_snr_min"],
        utterance_snr_max=_DEFAULTS["utterance_snr_max"],
    ):
        self.rate = rate
        self.mixing_noise_rate = mixing_noise_rate
        self.noise_shaper = NoiseAugmentTransform(noise_path)
        self.noise_snr_min = noise_snr_min
        self.noise_snr_max = noise_snr_max
        self.utterance_snr_min = utterance_snr_min
        self.utterance_snr_max = utterance_snr_max

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"rate={self.rate}",
                    f"mixing_noise_rate={self.mixing_noise_rate}",
                    f"noise_snr_min={self.noise_snr_min}",
                    f"noise_snr_max={self.noise_snr_max}",
                    f"utterance_snr_min={self.utterance_snr_min}",
                    f"utterance_snr_max={self.utterance_snr_max}",
                ]
            )
            + ")"
        )

    def __call__(self, sources):
        for i, source in enumerate(sources):
            if np.random.random() > self.rate:
                continue

            pri = source.numpy()

            if np.random.random() > self.mixing_noise_rate:
                sec = sources[np.random.randint(0, len(sources))].numpy()
                snr = rand_uniform(self.utterance_snr_min, self.utterance_snr_max)
            else:
                sec = self.noise_shaper.pick_sample(source.shape)
                snr = rand_uniform(self.noise_snr_min, self.noise_snr_max)

            L1 = pri.shape[-1]
            L2 = sec.shape[-1]
            l = np.random.randint(0, min(round(L1 / 2), L2))  # mix len
            s_source = np.random.randint(0, L1 - l)
            s_sec = np.random.randint(0, L2 - l)

            get_power = lambda x: np.mean(x**2)
            if get_power(sec) == 0:
                continue

            scl = np.sqrt(get_power(pri) / (np.power(10, snr / 10) * get_power(sec)))

            pri[s_source : s_source + l] = np.add(
                pri[s_source : s_source + l], np.multiply(scl, sec[s_sec : s_sec + l])
            )
            sources[i] = torch.from_numpy(pri).float()

        return sources
