# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import warnings


class Naive_F0_Decoder(torch.nn.Module):
    def __init__(self, bounds_path, n_units=32):
        super().__init__()

        bounds = torch.load(bounds_path)
        bounds = torch.from_numpy(bounds[n_units])
        assert bounds.ndim == 1

        pad = torch.tensor([-5.0, -5.0])  # bos, eos, pad are in the dictionary
        centers = torch.cat(
            [bounds[0:1], 0.5 * (bounds[1:] + bounds[:-1]), bounds[-1:], pad[:]]
        )

        self.embedding = torch.nn.Embedding.from_pretrained(
            centers.unsqueeze(-1), freeze=True
        )
        self.max_n = self.embedding.weight.numel()

    def forward(self, discrete_f0: torch.Tensor):
        in_bounds = (0 <= discrete_f0).all() and (discrete_f0 < self.max_n).all()
        if not in_bounds:
            warnings.warn(
                f"F0 contains some weird outputs: discrete_f0.max().item()={discrete_f0.max().item()} discrete_f0.min().item()={discrete_f0.min().item()}; "
                f"while we have embeddings for {self.max_n} values. "
                "Assuming this is a no-prosody model -- but be careful!"
            )

            mask = discrete_f0 >= self.max_n
            discrete_f0 = discrete_f0.masked_fill(mask, self.max_n - 1)

        return self.embedding(discrete_f0).squeeze(-1)
