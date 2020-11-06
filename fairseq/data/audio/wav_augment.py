import logging
import numpy as np
from typing import Callable, Tuple
import torch
import os
import numpy as np
import re
from functools import partial
from .. import BaseWrapperDataset

logger = logging.getLogger(__name__)
DBFS_COEF = 10.0 / torch.log10(torch.scalar_tensor(10.0))


class AdditiveNoise:
    """Use customized AdditiveNoise to replace EffectChain.additive_noise,
    because the EffectChain.additive_noise needs matched lengths between audio and noise,
    which is not easy to implement dynamic mixing,
    and the snr in EffectChain.additive_noise is constance and is not the definition of `Signal-to-Noise Ratio`,
    which doesn't calculate the power of signal between signal and noise,
    so the official implementation might cause the noise to overwhelm the signal if the volumes of datasets is not balanced.
    """

    def __init__(self, noise_dir: str, snr: Tuple[float, Callable]):
        self.snr = snr

        if noise_dir:
            assert os.path.exists(noise_dir)
        self.noise_dir = os.path.abspath(noise_dir) if noise_dir else None

        self.noise_files = []
        self.noise_files_len = 0
        import soundfile as sf
        self.sf = sf

        if self.noise_dir:
            for dirpath, dirnames, filenames in os.walk(self.noise_dir):
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    self.noise_files.append(path)
            self.noise_files_len = len(self.noise_files)

            assert self.noise_files_len > 0, \
                "the noise dir is empty: {}".format(noise_dir)

            logger.info(f"loaded {self.noise_files_len} noise samples from `{noise_dir}`")

    def wav_to_dbfs(self, x: torch.Tensor) -> torch.Tensor:
        return DBFS_COEF * torch.log10(torch.dot(x[0, :], x[0, :]) / x.shape[1] + 1e-8)

    def sample_noise(self) -> torch.Tensor:
        
        try:
            path = self.noise_files[np.random.randint(self.noise_files_len)]
            wav, curr_sample_rate = self.sf.read(path)
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

        noised = x + seg_noise * min(gain_ratio, max_gain_ratio)

        # peak protection
        compress_gain = 1.0 / torch.max(noised)
        if compress_gain < 1.0:
            noised *= compress_gain

        return noised, src_info, dst_info


class ChainRunner(object):
    """
    Takes an instance of augment.EffectChain and applies it on pytorch tensors.
    """

    def __init__(self, chain, sample_rate):
        self.chain = chain
        self.sample_rate = sample_rate

    def __call__(self, x):
        """
        x: torch.Tensor, (channels, length). Must be placed on CPU.
        """
        x = x.view(1, -1)
        src_info = {'channels': x.size(0),  # number of channels
                    'length': x.size(1),  # length of the sequence
                    'precision': 32,  # precision (16, 32 bits)
                    'rate': self.sample_rate,  # sampling rate
                    'bits_per_sample': 32}  # size of the sample

        target_info = {'channels': 1,
                       'length': x.size(1),
                       'precision': 32,
                       'rate': self.sample_rate,
                       'bits_per_sample': 32}

        y = self.chain.apply(x, src_info=src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x
        return y.squeeze(0)


class AudioAugmentDataset(BaseWrapperDataset):
    def __init__(self, dataset, wav_augment_commands):
        super().__init__(dataset)
        effect_chain = chain_factory(wav_augment_commands)
        self.runner = ChainRunner(effect_chain, dataset.sample_rate)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["original_source"] = item["source"]
        item["source"] = self.runner(item["source"])
        return item


def parse_value(value):
    if '~' in value:
        min_val, max_val = [parse_value(v) for v in value.split('~')]
        assert min_val != max_val, \
            "uniform mix/max value are the same: {}".format(min_val)
        assert type(min_val) == type(max_val), \
            "min_val and max_val should be the same datatype but got: {}({}) {}({})".format(
                min_val, type(min_val), max_val, type(max_val))
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        if isinstance(min_val, int):
            value = partial(np.random.randint, min_val, max_val)
        elif isinstance(min_val, float):
            value = partial(np.random.uniform, min_val, max_val)
        else:
            raise "{} parsing error".format(value)
    elif re.findall(r'^[0-9\-]+$', value):
        value = int(value)
    elif re.findall(r'^[\.0-9\-]+$', value):
        value = float(value)
    return value


def chain_factory(commands):
    '''To build the ChainEffect by parsing string command.

    Read the `https://github.com/facebookresearch/WavAugment` to understand how many effects we can use.
    And the source methods come from `http://sox.sourceforge.net/sox.html`

    Args:
        commands (str): the commands separated by `+` to describe your chain effect
        The commands structure should be: `<effect-name1>:arg1,arg2,arg3,key1=value1,key2=value2+<effect-name2>:arg1,arg2,arg3,key1=value1,key2=value2`
        For example: `speed:0.9~1.5+pitch:50+rate:16000+additive_noise:/path/to/noises_dir,5~20` means
                randomly change the speech between 0.9~1.5 -> change pitch to 50 cents -> keep framerate at 16000
                -> randomly mix noise into src audio, and randomly keep the snr between 5~20
    '''
    try:
        from augment import EffectChain
    except ImportError as err:
        err_msg = str(err)
        err_msg += "\nTo apply the wav_augment, you should install the `facebookresearch/WavAugment` first.\n" + \
            "Please read the installtion guide from `https://github.com/facebookresearch/WavAugment`"
        raise ImportError(err_msg)

    chain = EffectChain()
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
        if method_name == 'additive_noise':
            try:
                chain._chain.append(AdditiveNoise(*args, **kwargs))
            except Exception as err:
                err_msg = str(err)
                err_msg += "\nThe additive_noise effect is NOT the official one, you should specify the args in wav-augment like below:\n" + \
                    "\t`...+additive_noise:/the/noise/dir,5~30+...`"
                raise AssertionError(err_msg)
        else:
            method = getattr(chain, method_name)
            chain = method(*args, **kwargs)
    return chain
