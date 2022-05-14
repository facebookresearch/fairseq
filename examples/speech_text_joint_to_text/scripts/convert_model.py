#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import re
from collections import OrderedDict

import torch

from fairseq.file_io import PathManager


def is_update(param_name, module_name):
    if module_name in param_name:
        return True
    return False


def load_checkpoint(src_cpt):

    with PathManager.open(src_cpt, "rb") as f:
        state_src = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )

    return state_src


def save_checkpoint(tgt_cpt, states):

    with PathManager.open(tgt_cpt, "wb") as f:
        torch.save(
            states,
            f,
        )


# convert the pre-trained model into bart model
def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument('--input-model', required=True,
                        help='Input checkpoint file path.')
    parser.add_argument('--output-model', required=True,
                        help='output checkpoint file path.')
    # fmt: on
    args = parser.parse_args()
    print(args)

    states = load_checkpoint(args.input_model)
    model = states["model"]
    new_model = OrderedDict()
    for key in model.keys():
        if re.search("^encoder.text_encoder", key):
            new_key = re.sub("encoder.text_encoder", "encoder", key)
            new_model[new_key] = model[key]
        elif re.search("^decoder.text_decoder", key):
            new_key = re.sub("decoder.text_decoder", "decoder", key)
            new_model[new_key] = model[key]
    states["model"] = new_model
    save_checkpoint(args.output_model, states)


if __name__ == "__main__":
    main()
