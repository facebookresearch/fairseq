import glob
import os
from typing import List, Optional, Tuple

import logging
import numpy as np
import torchvision.transforms.functional as TF
import PIL
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger(__name__)


class PathDataset(VisionDataset):
    def __init__(
        self,
        root: List[str],
        loader: None = None,
        transform: Optional[str] = None,
        extra_transform: Optional[str] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super().__init__(root=root)

        PIL.Image.MAX_IMAGE_PIXELS = 256000001

        self.files = []
        for folder in self.root:
            self.files.extend(
                sorted(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True))
            )
            self.files.extend(
                sorted(glob.glob(os.path.join(folder, "**", "*.png"), recursive=True))
            )

        self.transform = transform
        self.extra_transform = extra_transform
        self.mean = mean
        self.std = std

        self.loader = loader

        logger.info(f"loaded {len(self.files)} samples from {root}")

        assert (mean is None) == (std is None)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        path = self.files[idx]

        if self.loader is not None:
            return self.loader(path), None

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        img = TF.to_tensor(img)
        if self.mean is not None and self.std is not None:
            img = TF.normalize(img, self.mean, self.std)
        return img, None
