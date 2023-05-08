#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch

from omegaconf import OmegaConf

from fairseq.criterions.model_criterion import ModelCriterionConfig
from fairseq.dataclass.configs import FairseqConfig

from tasks import ImageClassificationConfig, ImagePretrainingConfig
from models.data2vec_image_classification import (
    Data2VecImageClassificationConfig,
    Data2VecImageClassificationModel,
)
from models.data2vec_vision import Data2VecVisionConfig, Data2VecVisionModel


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert beit checkpoint into data2vec - vision checkpoint"
    )
    # fmt: off
    parser.add_argument('checkpoint', help='checkpoint to convert')
    parser.add_argument('--output', required=True, metavar='PATH', help='where to output converted checkpoint')
    parser.add_argument('--type', type=str, choices=['vision', 'image_classification'], default='image_classification', help='type of model to upgrade')
    parser.add_argument('--inception_norms', action='store_true', default=False)
    # fmt: on

    return parser


def update_checkpoint(model_dict, prefix, is_nested):

    replace_paths = {
        "cls_token": "model.cls_emb" if is_nested else "cls_emb",
        "patch_embed": "model.patch_embed" if is_nested else "patch_embed",
        "mask_token": "mask_emb",
    }

    starts_with = {
        "patch_embed.proj": "model.patch_embed.conv"
        if is_nested
        else "patch_embed.conv",
        "lm_head": "final_proj",
        "fc_norm": "fc_norm",
        "head": "head",
    }

    partial = {
        "mlp.fc1": "mlp.0",
        "mlp.fc2": "mlp.2",
    }

    for k in list(model_dict.keys()):
        for sw, r in starts_with.items():
            if k.startswith(sw):
                replace_paths[k] = k.replace(sw, r)
        for p, r in partial.items():
            if p in k:
                replace_paths[k] = prefix + k.replace(p, r)

    if prefix != "":
        for k in list(model_dict.keys()):
            if k not in replace_paths:
                replace_paths[k] = prefix + k

    for k in list(model_dict.keys()):
        if k in replace_paths:
            model_dict[replace_paths[k]] = model_dict[k]
            if k != replace_paths[k]:
                del model_dict[k]

    return model_dict


def main():
    parser = get_parser()
    args = parser.parse_args()

    cp = torch.load(args.checkpoint, map_location="cpu")

    cfg = FairseqConfig(
        criterion=ModelCriterionConfig(_name="model", log_keys=["correct"]),
    )

    if args.type == "image_classification":

        cfg.task = ImageClassificationConfig(
            _name="image_classification",
            data=".",
        )

        if args.inception_norms:
            cfg.task.normalization_mean = [0.5, 0.5, 0.5]
            cfg.task.normalization_std = [0.5, 0.5, 0.5]

        cfg.model = Data2VecImageClassificationConfig(
            _name="data2vec_image_classification",
        )
        cfg.model.pretrained_model_args = FairseqConfig(
            model=Data2VecVisionConfig(
                _name="data2vec_vision", shared_rel_pos_bias=False
            ),
            task=ImagePretrainingConfig(
                _name="image_pretraining",
            ),
        )

        cfg = OmegaConf.create(cfg)

        state = {
            "cfg": OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
            "model": cp["module"],
            "best_loss": None,
            "optimizer": None,
            "extra_state": {},
        }

        model = Data2VecImageClassificationModel(cfg.model)
        model.load_state_dict(
            update_checkpoint(state["model"], prefix="model.encoder.", is_nested=True),
            strict=True,
        )
    elif args.type == "vision":
        cfg.task = ImagePretrainingConfig(
            _name="image_pretraining",
            data=".",
        )

        if args.inception_norms:
            cfg.task.normalization_mean = [0.5, 0.5, 0.5]
            cfg.task.normalization_std = [0.5, 0.5, 0.5]

        cfg.model = Data2VecVisionConfig(
            _name="data2vec_vision",
        )
        cfg = OmegaConf.create(cfg)

        state = {
            "cfg": OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
            "model": cp["model"],
            "best_loss": None,
            "optimizer": None,
            "extra_state": {},
        }

        model = Data2VecVisionModel(cfg.model)
        model.load_state_dict(
            update_checkpoint(state["model"], prefix="encoder.", is_nested=False),
            strict=True,
        )
    else:
        raise Exception("unsupported type " + args.type)

    print(state["cfg"], state.keys())
    torch.save(state, args.output)


if __name__ == "__main__":
    main()
