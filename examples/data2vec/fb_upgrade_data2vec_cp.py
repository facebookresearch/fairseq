#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch


def get_parser():
    parser = argparse.ArgumentParser(description="convert data2vec checkpoint")
    # fmt: off
    parser.add_argument('checkpoint', help='checkpoint to convert')
    parser.add_argument('--output', required=True, metavar='PATH', help='where to output converted checkpoint')
    parser.add_argument('--type', type=str, choices=['audio', 'audio_ctc', 'nlp'], default='audio', help='type of model to upgrade')
    # fmt: on

    return parser


def upgrade_common(cfg):
    pass

def upgrade_audio(cfg):
    del cfg["task"]["cache_in_scratch"]
    cfg["model"]["_name"] = "data2vec_audio"
    cfg["task"]["normalize"] = True

def upgrade_audio_ctc(cfg):
    cfg["task"]["normalize"] = True
    upgrade_audio(cfg["model"]["w2v_args"])

def upgrade_nlp(cfg):
    cfg["model"]["transformer"] = {
        "layernorm_embedding": True,
        "activation_fn": "gelu",
        "no_scale_embedding": True,
        "encoder": {
            "embed_dim": 768,
            "ffn_embed_dim": 3072,
            "layers": 12,
            "attention_heads": 12,
            "normalize_before": False,
            "learned_pos": True,
            "layerdrop": 0,
        },
    }

def update_prefix(model_dict, prefix, new_prefix):
    for k in list(model_dict.keys()):
        if prefix and k.startswith(prefix):
            new_k = k.replace(prefix, new_prefix, 1)
            model_dict[new_k] = model_dict[k]
            del model_dict[k]


def update_checkpoint(model_dict, prefix=""):
    replace_paths = {
    }

    if prefix:
        replace_paths = {prefix + k: prefix + v for k, v in replace_paths.items()}

    for k in list(model_dict.keys()):
        if k in replace_paths:
            model_dict[replace_paths[k]] = model_dict[k]
            del model_dict[k]


def main():
    parser = get_parser()
    args = parser.parse_args()

    cp = torch.load(args.checkpoint, map_location="cpu")
    upgrade_common(cp["cfg"])

    if args.type == 'audio':
        upgrade_audio(cp["cfg"])
    elif args.type == 'audio_ctc':
        upgrade_audio_ctc(cp["cfg"])
    elif args.type == 'nlp':
        upgrade_nlp(cp["cfg"])

    print(cp["cfg"])
    torch.save(cp, args.output)


if __name__ == "__main__":
    main()
