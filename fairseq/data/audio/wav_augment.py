import numpy as np
from typing import Callable, Tuple
import torch
import soundfile as sf
import os
import torch
import numpy as np
import re

try:
    from augment import EffectChain
except ImportError as err:
    err_msg = str(err)
    err_msg += "\nTo apply the wav_augment, you should install the `facebookresearch/WavAugment` first.\n" + \
        "Please read the installtion guide from `https://github.com/facebookresearch/WavAugment`"
    raise ImportError(err_msg)


DBFS_COEF = 10.0 / torch.log10(torch.scalar_tensor(10.0))


class AdditiveNoise:
    def __init__(self, noise_dir: str, snr: Tuple[float, Callable]):
        self.snr = snr

        if noise_dir:
            assert os.path.exists(noise_dir)
        self.noise_dir = os.path.abspath(noise_dir) if noise_dir else None

        self.noise_files = []
        self.noise_files_len = 0

        if self.noise_dir:
            for dirpath, dirnames, filenames in os.walk(self.noise_dir):
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    self.noise_files.append(path)
            self.noise_files_len = len(self.noise_files)

            assert self.noise_files_len > 0, \
                "the noise dir is empty: {}".format(noise_dir)

    def wav_to_dbfs(self, x: torch.Tensor) -> torch.Tensor:
        return DBFS_COEF * torch.log10(torch.dot(x[0, :], x[0, :]) / x.shape[1] + 1e-8)

    def sample_noise(self) -> torch.Tensor:
        try:
            path = self.noise_files[np.random.randint(self.noise_files_len)]
            wav, curr_sample_rate = sf.read(path)
            return torch.from_numpy(wav).float().unsqueeze(0)
        except Exception as err:
            print(err)
            return self.sample_noise()

    def __call__(self, x, src_info, dst_info):
        noise = self.sample_noise()
        noise_len = noise.shape[1]
        src_len = x.shape[1]

        max_gain_ratio = 1.0 / torch.max(noise)

        noise = noise.repeat((1, np.math.ceil(src_len / noise_len)))
        noise_len = noise.shape[1]
        delta_len = noise_len - src_len

        start = np.random.randint(delta_len) if delta_len > 0 else 0
        end = start + src_len

        seg_noise = noise[:, start:end]

        feats_dbfs = self.wav_to_dbfs(x)
        seg_dbfs = self.wav_to_dbfs(seg_noise)
        current_snr = feats_dbfs - seg_dbfs
        target_snr = self.snr() if callable(self.snr) else self.snr
        gain_db = current_snr - target_snr
        gain_ratio = torch.pow(10, gain_db / 20)

        noised = x + seg_noise * gain_ratio

        # peak protection
        compress_gain = 1.0 / torch.max(noised)
        if compress_gain < 1.0:
            noised *= compress_gain

        return noised, src_info, dst_info


class CustomEffectChain(EffectChain):
    """To override the EffectChain.additive_noise, due to randomly choosing the snr,
    and automatically match the sample length between audio and noise
    """
    def additive_noise(self, noise_dir: str, snr: Tuple[float, Callable]):
        self._chain.append(AdditiveNoise(
            noise_dir=noise_dir, snr=snr))
        return self


def parse_value(value):
    if '~' in value:
        min_val, max_val = [parse_value(v) for v in value.split('~')]
        assert min_val != max_val, \
            "uniform mix/max value are the same: {}".format(min_val)
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        value = lambda: np.random.uniform(min_val, max_val)
    elif value.isnumeric():
        value = int(value)
    elif re.findall(r'^[\.0-9]+$', value):
        value = float(value)
    return value


def chain_fatory(commands) -> CustomEffectChain:
    '''To build the ChainEffect by parsing string command.

    Read the `https://github.com/facebookresearch/WavAugment` to understand how many effects we can use.

    Args:
        commands (str): the commands separated by `+` to describe your chain effect
        The commands structure should be: `<effect-name1>:arg1,arg2,arg3,key1=value1,key2=value2+<effect-name2>:arg1,arg2,arg3,key1=value1,key2=value2`
        For example: `speed:0.9~1.5+pitch:1.1+rate:16000+additive_noise:/path/to/noises_dir,5~20` means
                randomly change the speech between 0.9~1.5 -> change pitch to 1.1 -> keep framerate at 16000
                -> randomly mix noise into src audio, and randomly keep the snr between 5~20
    '''
    chain = CustomEffectChain()
    for command in commands.strip('+').split('+'):
        if not command:
            continue
        method_name, params = command.split(':')
        params = params.split(',')
        args = []
        kwargs = {}
        for i, param in enumerate(params):
            if '=' in param:
                key, value = param.split('=')
                kwargs[key] = parse_value(value)
            else:
                args.append(parse_value(param))
        method = getattr(chain, method_name)
        chain = method(*args, **kwargs)
    return chain
