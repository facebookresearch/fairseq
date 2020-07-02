#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import language_technology.neural_mt.vocab.constants as vocab_constants
import torch
from fairseq import checkpoint_utils, utils
from fairseq.file_io import PathManager

# Register latte bucket
from fblearner.flow.projects.fairseq.latte_training import manifold_file_io  # noqa


logger = logging.getLogger(__name__)


OSS_TO_FB_TASK_MAP = {
    "translation": "fb_translation",
    "translation_from_pretrained_bart": "fb_translation_from_pretrained_bart",
}

OSS_SPECIAL_TOKENS = 4


def convert_model_to_fb_dict(checkpoint, output_path):

    logger.info(f"Loading checkpoint {checkpoint}...")
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
    logger.info("Loaded checkpoint.")

    for parameter_name in state["model"]:
        if parameter_name.endswith("embed_tokens.weight"):
            oss_param = state["model"][parameter_name]
            oss_embed_count, embed_dim = oss_param.shape
            fb_param = torch.zeros(
                (
                    oss_embed_count
                    + vocab_constants.MAX_SPECIAL_TOKENS
                    - OSS_SPECIAL_TOKENS,
                    embed_dim,
                )
            )

            # Common reserved
            fb_param[:OSS_SPECIAL_TOKENS, :] = oss_param[:OSS_SPECIAL_TOKENS, :]

            # Remapped params
            fb_param[vocab_constants.MAX_SPECIAL_TOKENS:, :] = oss_param[
                OSS_SPECIAL_TOKENS:, :
            ]

            state["model"][parameter_name] = fb_param
        elif parameter_name.endswith("embed_positions.weight"):
            oss_param = state["model"][parameter_name]
            state["model"][parameter_name] = oss_param[: len(oss_param) - 1, :]

    if hasattr(state["args"], "task"):
        if state["args"].task in OSS_TO_FB_TASK_MAP:
            state["args"].task = OSS_TO_FB_TASK_MAP[state["args"].task]
        else:
            logger.warning(
                f"Task {state['args'].task} has no FB equivalent! Leaving in place."
            )

    state = utils.move_to_cpu(state)

    with PathManager.open(output_path, "wb") as f:
        checkpoint_utils.torch_persistent_save(state, f)
        logger.info(f"Wrote new checkpoint to {output_path}")


def main():
    """This script converts an OSS fairseq model to an internal model that can
    be used with the fb_translation or fb_translation_from_pretrained_bart tasks.

    Usage:
    buck run \
    //deeplearning/projects/fairseq-py/examples/fb_latte_prod:convert_model_to_fb_dict \
    -- --source [SOURCE] --destination [DESTINATION]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="path to source model (OSS Fairseq Model), example: "
        "manifold://latte_training/tree/mbart/mbart25.pt",
    )

    parser.add_argument(
        "--destination",
        required=True,
        help="path to destination for internal fb model, example: "
        "manifold://latte_training/tree/adityapillai/mbart_pretraining/mbart25_fb.pt",
    )

    args = parser.parse_args()
    convert_model_to_fb_dict(args.source, args.destination)


if __name__ == "__main__":
    main()
