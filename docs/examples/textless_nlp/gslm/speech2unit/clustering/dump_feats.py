# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

from examples.textless_nlp.gslm.speech2unit.pretrained.utils import (
    get_and_dump_features,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute and dump log mel fbank features."
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default=None,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--out_features_path",
        type=str,
        default=None,
        help="Features file path to write to",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Pretrained acoustic model checkpoint",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--sample_pct",
        type=float,
        help="Percent data to use for K-means training",
        default=0.1,
    )
    parser.add_argument(
        "--out_features_path",
        type=str,
        help="Path to save log mel fbank features",
    )
    return parser


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    """
    Example command:
    python ~/speechbot/clustering/dump_logmelfank_feats.py \
        --manifest_path /checkpoint/kushall/data/LJSpeech-1.1/asr_input_wavs_16k/train.tsv
        --out_features_path /checkpoint/kushall/experiments/speechbot/logmelfbank/features/ljspeech/train.npy
    """
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)

    logger.info(f"Extracting {args.feature_type} acoustic features...")
    get_and_dump_features(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint_path,
        layer=args.layer,
        manifest_path=args.manifest_path,
        sample_pct=args.sample_pct,
        flatten=True,
        out_features_path=args.out_features_path,
    )
    logger.info(f"Saved extracted features at {args.out_features_path}")
