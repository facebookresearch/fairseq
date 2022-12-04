#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch


def get_parser():
    parser = argparse.ArgumentParser(description="convert wav2vec checkpoint")
    # fmt: off
    parser.add_argument('checkpoint', help='checkpoint to convert')
    parser.add_argument('--output', required=True, metavar='PATH', help='where to output converted checkpoint')
    parser.add_argument('--type', type=str, choices=['wav2vec2', 'ctc', 'seq2seq'], default='wav2vec2', help='type of model to upgrade')
    # fmt: on

    return parser


def upgrade_common(args):
    args.task = "audio_pretraining"
    args.criterion = "wav2vec"
    args.normalize = getattr(args, "normalize", False)


def upgrade_bert_args(args):
    upgrade_common(args)
    args.arch = "wav2vec2"
    args.final_dim = args.mlp_mi
    args.latent_groups = args.latent_var_banks
    args.extractor_mode = "layer_norm" if args.normalize else "default"

    if hasattr(args, "latent_dim"):
        del args.latent_dim


def update_prefix(model_dict, prefix, new_prefix):
    for k in list(model_dict.keys()):
        if prefix and k.startswith(prefix):
            new_k = k.replace(prefix, new_prefix, 1)
            model_dict[new_k] = model_dict[k]
            del model_dict[k]


def update_checkpoint(model_dict, prefix=""):
    replace_paths = {
        "post_concat_proj.weight": "post_extract_proj.weight",
        "post_concat_proj.bias": "post_extract_proj.bias",
        "encoder.mask_emb": "mask_emb",
        "encoder.emb_layer_norm.weight": "encoder.layer_norm.weight",
        "encoder.emb_layer_norm.bias": "encoder.layer_norm.bias",
        "encoder.final_proj.weight": "final_proj.weight",
        "encoder.final_proj.bias": "final_proj.bias",
        "encoder.quantizer.vars": "quantizer.vars",
        "encoder.quantizer.weight_proj.weight": "quantizer.weight_proj.weight",
        "encoder.quantizer.weight_proj.bias": "quantizer.weight_proj.bias",
        "encoder.project_q.weight": "project_q.weight",
        "encoder.project_q.bias": "project_q.bias",
        "encoder.layer_norm_repr.weight": "encoder.layer_norm.weight",
        "encoder.layer_norm_repr.bias": "encoder.layer_norm.bias",
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
    upgrade_common(cp["args"])

    if args.type == "ctc":
        cp["args"].arch = "wav2vec_ctc"
        bert_path = getattr(cp["args"], 'bert_path', getattr(cp["args"], 'w2v_path', None))

        if not os.path.exists(bert_path):
            bert_path = "/checkpoint/abaevski/asr/speechbert_raw_big/spb_librspeech_big_v2.qtz.mlp768.pq.lv320.lvb2.ab0.9_0.98.lr0.0003.wu20000.mask10.mprob0.65.mstd0.mpl15.drp_i0.2.drp_f0.2.in0.0.nt_gaus.nnf0.ng512.fgm0.1.nep.qini.qini1.pen0_0_0.1_10.cpl1.ld0.1.uf1.mu250000.s5.ngpu128/checkpoint_best.pt"

        bcp = torch.load(bert_path, map_location="cpu")
        upgrade_bert_args(bcp["args"])
        cp["args"].w2v_args = bcp["args"]

        if hasattr(cp["args"], "bert_path"):
            del cp["args"].bert_path

        update_prefix(cp["model"], "bert", "w2v_encoder")
        update_prefix(cp["model"], "w2v_encoder.bert_model", "w2v_encoder.w2v_model")
        update_checkpoint(cp["model"], "w2v_encoder.w2v_model.")
    elif args.type == "seq2seq":
        cp["args"].arch = "wav2vec_seq2seq"
        bert_path = cp["args"].bert_path

        cp["args"].decoder_dropout = cp["args"].dropout
        cp["args"].decoder_attention_dropout = cp["args"].attention_dropout
        cp["args"].decoder_activation_dropout = cp["args"].activation_dropout

        bcp = torch.load(bert_path, map_location="cpu")
        upgrade_bert_args(bcp["args"])
        cp["args"].w2v_args = bcp["args"]
        del cp["args"].bert_path

        update_prefix(cp["model"], "encoder.bert_model", "encoder.w2v_model")
        update_checkpoint(cp["model"], "encoder.w2v_model.")
        del cp["model"]["decoder.version"]
    else:
        upgrade_bert_args(cp["args"])
        update_checkpoint(cp["model"])

    print(cp["args"])
    torch.save(cp, args.output)


if __name__ == "__main__":
    main()
