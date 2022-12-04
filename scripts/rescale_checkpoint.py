#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from fairseq.file_io import PathManager


def rescale_weights(weight_dict, scale):
    for k, v in weight_dict.items():
        if isinstance(v, dict):
            rescale_weights(v, scale)
        else:
            if not torch.is_tensor(v) or not torch.is_floating_point(v):
                print("skipping", k)
            else:
                weight_dict[k] = v * scale


def main():
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
        "produce a new checkpoint",
    )
    # fmt: off
    parser.add_argument('--input', required=True, metavar='FILE',
                        help='Input checkpoint file path.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    parser.add_argument('--scale', required=True, type=float, help='what to multiply each checkpoint weight by')
    # fmt: on
    args = parser.parse_args()
    print(args)

    with PathManager.open(args.input, "rb") as f:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )

    rescale_weights(state["model"], args.scale)
    with PathManager.open(args.output, "wb") as f:
        torch.save(state, f)
    print("Finished writing rescaled checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
