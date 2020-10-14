import numpy as np
from fairseq.data.audio.feature_transforms import (
    AudioFeatureTransform, register_audio_feature_transform
)


@register_audio_feature_transform('global_cmvn')
class GlobalCMVN(AudioFeatureTransform):
    """Global CMVN (cepstral mean and variance normalization). The global mean
    and variance need to be pre-computed and stored in NumPy format (.npz)."""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return GlobalCMVN(_config.get('stats_npz_path'))

    def __init__(self, stats_npz_path):
        stats = np.load(stats_npz_path)
        self.mean, self.std = stats['mean'], stats['std']

    def __call__(self, x):
        x = np.subtract(x, self.mean)
        x = np.divide(x, self.std)
        return x
