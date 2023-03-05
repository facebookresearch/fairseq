import torch
from torch import Tensor

from fairseq.data.audio.feature_transforms import (
    AudioFeatureTransform,
    register_audio_feature_transform,
)

def warp_axis_torch(specgram: Tensor, axis: int, W: float):
    """
    Warp axis (frequency or time) between W boundaries, starting from point w0, the warp 
    direction can be negative or positive, depending on the randomly chosen distance w.
    Args:
        specgram: A tensor with dimensions (batch, freq, time)
        axis: Axis where the warp takes place
        W: Boundary of time steps where the warp takes place (W, num_warped_axis - W)
    Returns:
        Tensor: Warped spectrogram of dimensions (batch, freq, time)
    """

    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    num_warped = specgram.shape[axis]
    num_non_warped = specgram.shape[1 if axis == 2 else 2]

    if W == 0:
        return specgram
    assert 2 * W < num_warped, (
        f"Warp param (W) {W} must be smaller than half the size of the warped axis {num_warped}")

    w0 = torch.randint(W, num_warped - W, ())
    w = torch.randint(-W + 1, W, ())

    if axis == 1:
        lower, upper = specgram[:, :w0, :], specgram[:, w0:, :]
        lower_sz = (w0 + w, num_non_warped)
        upper_sz = (num_warped - w0 - w, num_non_warped)
    else:
        lower, upper = specgram[:, :, :w0], specgram[:, :, w0:]
        lower_sz = (num_non_warped, w0 + w)
        upper_sz = (num_non_warped, num_warped - w0 - w)

    # interpolate receives 4D: (batch, channel, freq, time)
    lower = lower.unsqueeze(1)
    upper = upper.unsqueeze(1)

    lower = torch.nn.functional.interpolate(
        lower, size=lower_sz, mode='bilinear')
    upper = torch.nn.functional.interpolate(
        upper, size=upper_sz, mode='bilinear')

    lower.squeeze_(1)
    upper.squeeze_(1)

    specgram = torch.cat((lower, upper), axis=axis)
    return specgram


def mask_along_axis(
    specgram: Tensor,
    axis: int,
    num_masks: int,
    mask_param: int,
    p: float = 0.0,
    mask_value: float = 0.0
):
    """
    Apply mask along a spectrogram.
    The length of the mask is randomly chosen, with a cap on mask_param.
    Args
        specgram: A tensor with dimensions (batch, freq, time)
        axis: Masking is applied (freq -> 1, time -> 2)
        num_masks: Number of masks
        mask_param: Max length allowed for each individual mask.
        p: Max proportion of masked rows/cols for each individual mask.
        mask_value: Value for the masked portions
    Returns
        Tensor: Masked spectrogram of dimensions (batch, freq, time)
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    mask_param = min(mask_param, int(specgram.shape[axis] * p))
    if mask_param < 1:
        return specgram

    mask_size = torch.randint(mask_param, ())

    for _ in range(num_masks):
        mask_start = torch.randint(specgram.shape[axis] - mask_size, ())
        mask_end = mask_start + mask_size

        if axis == 1:
            specgram[:, mask_start: mask_end, :] = mask_value
        else:
            specgram[:, :, mask_start: mask_end] = mask_value

    return specgram


def spec_augment(
    specgram: Tensor,
    warp_axis: int = 1,
    warp_param: int = 0,
    freq_mask_n: int = 0,
    freq_mask_param: int = 0,
    freq_mask_p: float = 0.0,
    time_mask_n: int = 0,
    time_mask_param: int = 0,
    time_mask_p: float = 0.0,
    mask_value: float = 0.0
):
    """
    SpecAugment to spectrogram with dimensions (batch, frequency, time)
    Args
        specgram: Tensor with dimensions (batch, frequency, time)
        warp_axis: Axis where the warp takes place (0->freq, 1->time)
        warp_param: Boundaries where warp takes place (W, N - W), (W in paper)
        freq_mask_n: Number of masks to apply to the frequency axis, (mF in paper)
        freq_mask_param: Max length of any individual frequency mask, (F in paper)
        freq_mask_p: Max proportion that any individual freq mask can have
        time_mask_n: Number of masks to apply to the time axis, (mT in paper)
        time_mask_param: Max length of any individual time mask, (T in paper)
        time_mask_p: Max proportion that any individual time mask can have, (p in paper)
    Returns
        Tensor: Augmented spectrogram with dimensions (batch, frequency, time)
    """

    specgram = specgram.clone()

    if specgram.dim() == 2:
        specgram = specgram.unsqueeze_(0)

    specgram = warp_axis_torch(specgram, warp_axis, warp_param)
    specgram = mask_along_axis(
        specgram, 1, freq_mask_n, freq_mask_param, freq_mask_p, mask_value)
    specgram = mask_along_axis(
        specgram, 2, time_mask_n, time_mask_param, time_mask_p, mask_value)
    return specgram


@register_audio_feature_transform("specaugment_torch")
class SpecAugmentTransformTorch(AudioFeatureTransform):
    """
    SpecAugment (https://arxiv.org/abs/1904.08779)
    Args
        warp_axis: Axis where the warp takes place (0->freq, 1->time)
        warp_param: Boundaries where warp takes place (W, N - W), (W in paper)
        freq_mask_n: Number of masks to apply to the frequency axis, (mF in paper)
        freq_mask_param: Max length of any individual frequency mask, (F in paper)
        freq_mask_p: Max proportion that any individual freq mask can have
        time_mask_n: Number of masks to apply to the time axis, (mT in paper)
        time_mask_param: Max length of any individual time mask, (T in paper)
        time_mask_p: Max proportion that any individual time mask can have, (p in paper)
    """

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return cls(
            _config.get("warp_axis", 2),
            _config.get("warp_param", 0),
            _config.get("freq_mask_n", 0),
            _config.get("freq_mask_param", 0),
            _config.get("freq_mask_p", 0.0),
            _config.get("time_mask_n", 0),
            _config.get("time_mask_param", 0),
            _config.get("time_mask_p", 0.0),
            _config.get("mask_value", 0.0),
        )
    
    def __init__(
        self,
        warp_axis: int = 2,
        warp_param: int = 0,
        freq_mask_n: int = 0,
        freq_mask_param: int = 0,
        freq_mask_p: float = 1.0,
        time_mask_n: int = 0,
        time_mask_param: int = 0,
        time_mask_p: float = 1.0,
        mask_value: float = 0.0

    ) -> None:
        super(SpecAugmentTransformTorch, self).__init__()
        self.warp_axis = warp_axis
        self.warp_param = warp_param
        self.freq_mask_n = freq_mask_n
        self.freq_mask_param = freq_mask_param
        self.freq_mask_p = freq_mask_p
        self.time_mask_n = time_mask_n
        self.time_mask_param = time_mask_param
        self.time_mask_p = time_mask_p
        self.mask_value = mask_value

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"warp_param={self.warp_param}",
                    f"freq_mask_n={self.freq_mask_n}",
                    f"freq_mask_param={self.freq_mask_param}",
                    f"freq_mask_p={self.freq_mask_p}",
                    f"time_mask_n={self.time_mask_n}",
                    f"time_mask_param={self.time_mask_param}",
                    f"time_mask_p={self.time_mask_p}",
                ]
            )
            + ")"
        )

    def __call__(self, specgram: torch.Tensor):
        """
        Args 
            specgram: Tensor with dimensions (batch, frequency, time)
        Returns
            specgram: Augmented specgram tensor with dimensions (batch, frequency, time)
        """
        return spec_augment(specgram, self.warp_axis, self.warp_param,
                            self.freq_mask_n, self.freq_mask_param, self.freq_mask_p,
                            self.time_mask_n, self.time_mask_param, self.time_mask_p,
                            self.mask_value)

    def libri_speech_basic(self):
        self.warp_param = 80
        self.freq_mask_param, self.freq_mask_n = 27, 1
        self.time_mask_param, self.time_mask_n = 100, 1
        self.time_mask_p = 1.0

    def libri_speech_double(self):
        self.warp_param = 80
        self.freq_mask_param, self.freq_mask_n = 27, 2
        self.time_mask_param, self.time_mask_n = 100, 2
        self.time_mask_p = 1.0

    def switchboard_mild(self):
        self.warp_param = 40
        self.freq_mask_param, self.freq_mask_n = 15, 2
        self.time_mask_param, self.time_mask_n = 70, 2
        self.time_mask_p = 0.2

    def switchboard_strong(self):
        self.warp_param = 40
        self.freq_mask_param, self.freq_mask_n = 27, 2
        self.time_mask_param, self.time_mask_n = 70, 2
        self.time_mask_p = 0.2
