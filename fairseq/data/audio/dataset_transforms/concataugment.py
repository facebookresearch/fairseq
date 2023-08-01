from typing import List
import numpy as np

from fairseq.data.audio.dataset_transforms import (
    AudioDatasetTransform,
    register_audio_dataset_transform,
)

_DEFAULTS = {"rate": 0.25, "max_tokens": 3000, "attempts": 5}


@register_audio_dataset_transform("concataugment")
class ConcatAugment(AudioDatasetTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return ConcatAugment(
            _config.get("rate", _DEFAULTS["rate"]),
            _config.get("max_tokens", _DEFAULTS["max_tokens"]),
            _config.get("attempts", _DEFAULTS["attempts"]),
        )

    def __init__(
        self,
        rate=_DEFAULTS["rate"],
        max_tokens=_DEFAULTS["max_tokens"],
        attempts=_DEFAULTS["attempts"],
    ):
        self.rate, self.max_tokens, self.attempts = rate, max_tokens, attempts

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"rate={self.rate}",
                    f"max_tokens={self.max_tokens}",
                    f"attempts={self.attempts}",
                ]
            )
            + ")"
        )

    def find_indices(self, index: int, n_frames: List[int], n_samples: int):
        # skip conditions: application rate, max_tokens limit exceeded
        if np.random.random() > self.rate:
            return [index]
        if self.max_tokens and n_frames[index] > self.max_tokens:
            return [index]

        # pick second sample to concatenate
        for _ in range(self.attempts):
            index2 = np.random.randint(0, n_samples)
            if index2 != index and (
                not self.max_tokens
                or n_frames[index] + n_frames[index2] < self.max_tokens
            ):
                return [index, index2]

        return [index]
