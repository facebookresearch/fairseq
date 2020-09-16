import numpy as np
import pickle as pkl
from fairseq.data.audio.transforms import S2TTransform, register_transform

@register_transform('global_cmvn')
class GlobalCMVN(S2TTransform):
    @classmethod
    def from_config_dict(cls, config):
        return GlobalCMVN(
            config.get('gcmvn'),
        )


    def __init__(self, gcmvn):
        with open(gcmvn, 'rb') as f:
            cmvn = pkl.load(f)
        self.mean = cmvn['mean']
        self.std = cmvn['std']


    def __call__(self, x):
        x = np.subtract(x, self.mean)
        x = np.divide(x, self.std)
        return x
