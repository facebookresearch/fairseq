from pathlib import Path
import numpy as np
from math import ceil

from fairseq.data.audio import rand_uniform
from fairseq.data.audio.waveform_transforms import (
    AudioWaveformTransform,
    register_audio_waveform_transform,
)

SNR_MIN = 5.0
SNR_MAX = 15.0
RATE = 0.25

NOISE_RATE = 1.0
NOISE_LEN_MEAN = 0.2
NOISE_LEN_STD = 0.05


class NoiseAugmentTransform(AudioWaveformTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return cls(
            _config.get("samples_path", None),
            _config.get("snr_min", SNR_MIN),
            _config.get("snr_max", SNR_MAX),
            _config.get("rate", RATE),
        )

    def __init__(
        self,
        samples_path: str,
        snr_min: float = SNR_MIN,
        snr_max: float = SNR_MAX,
        rate: float = RATE,
    ):
        # Sanity checks
        assert (
            samples_path
        ), "need to provide path to audio samples for noise augmentation"
        assert snr_max >= snr_min, f"empty signal-to-noise range ({snr_min}, {snr_max})"
        assert rate >= 0 and rate <= 1, "rate should be a float between 0 to 1"

        self.paths = list(Path(samples_path).glob("**/*.wav"))  # load music
        self.n_samples = len(self.paths)
        assert self.n_samples > 0, f"no audio files found in {samples_path}"

        self.snr_min = snr_min
        self.snr_max = snr_max
        self.rate = rate

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"n_samples={self.n_samples}",
                    f"snr={self.snr_min}-{self.snr_max}dB",
                    f"rate={self.rate}",
                ]
            )
            + ")"
        )

    def pick_sample(self, goal_shape, always_2d=False, use_sample_rate=None):
        from fairseq.data.audio.audio_utils import get_waveform

        path = self.paths[np.random.randint(0, self.n_samples)]
        sample = get_waveform(
            path, always_2d=always_2d, output_sample_rate=use_sample_rate
        )[0]

        # Check dimensions match, else silently skip adding noise to sample
        # NOTE: SHOULD THIS QUIT WITH AN ERROR?
        is_2d = len(goal_shape) == 2
        if len(goal_shape) != sample.ndim or (
            is_2d and goal_shape[0] != sample.shape[0]
        ):
            return np.zeros(goal_shape)

        # Cut/repeat sample to size
        len_dim = len(goal_shape) - 1
        n_repeat = ceil(goal_shape[len_dim] / sample.shape[len_dim])
        repeated = np.tile(sample, [1, n_repeat] if is_2d else n_repeat)
        start = np.random.randint(0, repeated.shape[len_dim] - goal_shape[len_dim] + 1)
        return (
            repeated[:, start : start + goal_shape[len_dim]]
            if is_2d
            else repeated[start : start + goal_shape[len_dim]]
        )

    def _mix(self, source, noise, snr):
        get_power = lambda x: np.mean(x**2)
        if get_power(noise):
            scl = np.sqrt(
                get_power(source) / (np.power(10, snr / 10) * get_power(noise))
            )
        else:
            scl = 0
        return 1 * source + scl * noise

    def _get_noise(self, goal_shape, always_2d=False, use_sample_rate=None):
        return self.pick_sample(goal_shape, always_2d, use_sample_rate)

    def __call__(self, source, sample_rate):
        if np.random.random() > self.rate:
            return source, sample_rate

        noise = self._get_noise(
            source.shape, always_2d=True, use_sample_rate=sample_rate
        )

        return (
            self._mix(source, noise, rand_uniform(self.snr_min, self.snr_max)),
            sample_rate,
        )


@register_audio_waveform_transform("musicaugment")
class MusicAugmentTransform(NoiseAugmentTransform):
    pass


@register_audio_waveform_transform("backgroundnoiseaugment")
class BackgroundNoiseAugmentTransform(NoiseAugmentTransform):
    pass


@register_audio_waveform_transform("babbleaugment")
class BabbleAugmentTransform(NoiseAugmentTransform):
    def _get_noise(self, goal_shape, always_2d=False, use_sample_rate=None):
        for i in range(np.random.randint(3, 8)):
            speech = self.pick_sample(goal_shape, always_2d, use_sample_rate)
            if i == 0:
                agg_noise = speech
            else:  # SNR scaled by i (how many noise signals already in agg_noise)
                agg_noise = self._mix(agg_noise, speech, i)
        return agg_noise


@register_audio_waveform_transform("sporadicnoiseaugment")
class SporadicNoiseAugmentTransform(NoiseAugmentTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return cls(
            _config.get("samples_path", None),
            _config.get("snr_min", SNR_MIN),
            _config.get("snr_max", SNR_MAX),
            _config.get("rate", RATE),
            _config.get("noise_rate", NOISE_RATE),
            _config.get("noise_len_mean", NOISE_LEN_MEAN),
            _config.get("noise_len_std", NOISE_LEN_STD),
        )

    def __init__(
        self,
        samples_path: str,
        snr_min: float = SNR_MIN,
        snr_max: float = SNR_MAX,
        rate: float = RATE,
        noise_rate: float = NOISE_RATE,  # noises per second
        noise_len_mean: float = NOISE_LEN_MEAN,  # length of noises in seconds
        noise_len_std: float = NOISE_LEN_STD,
    ):
        super().__init__(samples_path, snr_min, snr_max, rate)
        self.noise_rate = noise_rate
        self.noise_len_mean = noise_len_mean
        self.noise_len_std = noise_len_std

    def _get_noise(self, goal_shape, always_2d=False, use_sample_rate=None):
        agg_noise = np.zeros(goal_shape)
        len_dim = len(goal_shape) - 1
        is_2d = len(goal_shape) == 2

        n_noises = round(self.noise_rate * goal_shape[len_dim] / use_sample_rate)
        start_pointers = [
            round(rand_uniform(0, goal_shape[len_dim])) for _ in range(n_noises)
        ]

        for start_pointer in start_pointers:
            noise_shape = list(goal_shape)
            len_seconds = np.random.normal(self.noise_len_mean, self.noise_len_std)
            noise_shape[len_dim] = round(max(0, len_seconds) * use_sample_rate)
            end_pointer = start_pointer + noise_shape[len_dim]
            if end_pointer >= goal_shape[len_dim]:
                continue

            noise = self.pick_sample(noise_shape, always_2d, use_sample_rate)
            if is_2d:
                agg_noise[:, start_pointer:end_pointer] = (
                    agg_noise[:, start_pointer:end_pointer] + noise
                )
            else:
                agg_noise[start_pointer:end_pointer] = (
                    agg_noise[start_pointer:end_pointer] + noise
                )

        return agg_noise
