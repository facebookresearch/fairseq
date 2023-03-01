# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .image_pretraining import ImagePretrainingTask, ImagePretrainingConfig
from .image_classification import ImageClassificationTask, ImageClassificationConfig
from .mae_image_pretraining import MaeImagePretrainingTask, MaeImagePretrainingConfig


__all__ = [
    "ImageClassificationTask",
    "ImageClassificationConfig",
    "ImagePretrainingTask",
    "ImagePretrainingConfig",
    "MaeImagePretrainingTask",
    "MaeImagePretrainingConfig",
]