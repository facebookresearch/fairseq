import numpy as np

from fairseq.data.audio.transforms import S2TTransform, register_transform


@register_transform('utterance_cmvn')
class UtteranceCMVN(S2TTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return UtteranceCMVN(
            _config.get('norm_means', True),
            _config.get('norm_vars', True),
        )

    def __init__(self, norm_means=True, norm_vars=True):
        self.norm_means, self.norm_vars = norm_means, norm_vars

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(norm_means={self.norm_means}, norm_vars={self.norm_vars})'

    def __call__(self, x):
        square_sums = (x ** 2).sum(axis=0)
        mean = x.mean(axis=0)

        if self.norm_means:
            x = np.subtract(x, mean)
        if self.norm_vars:
            var = square_sums / x.shape[0] - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)

        return x
