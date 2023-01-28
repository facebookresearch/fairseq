# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import numpy as np

import joblib
from examples.textless_nlp.gslm.speech2unit.clustering.utils import (
    get_audio_files,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.utils import (
    get_features,
)


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quantize using K-means clustering over acoustic features."
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default=None,
        required=True,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--acoustic_model_path",
        type=str,
        help="Pretrained acoustic model checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--kmeans_model_path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Features file path. You don't need to enter acoustic model details if you have dumped features",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--out_quantized_file_path",
        required=True,
        type=str,
        help="File path of quantized output.",
    )
    parser.add_argument(
        "--extension", type=str, default=".flac", help="Features file path"
    )
    parser.add_argument(
        "--channel_id",
        choices=['1', '2'],
        help="The audio channel to extract the units in case of stereo file.",
        default=None,
    )
    parser.add_argument(
        "--hide-fname", action='store_true',
        help="Hide file names in the output file."
    )
    return parser


def main(args, logger):
    # Feature extraction
    if args.features_path is not None:
        logger.info(f"Loading acoustic features from {args.features_path}...")
        features_batch = np.load(args.features_path)
    else:
        logger.info(f"Extracting {args.feature_type} acoustic features...")
        features_batch = get_features(
            feature_type=args.feature_type,
            checkpoint_path=args.acoustic_model_path,
            layer=args.layer,
            manifest_path=args.manifest_path,
            sample_pct=1.0,
            flatten=False,
            channel_id=int(args.channel_id) if args.channel_id else None,
        )
        logger.info(
            f"Features extracted for {len(features_batch)} utterances.\n"
        )
        logger.info(
            f"Dimensionality of representation = {features_batch[0].shape[1]}"
        )

    # K-means model
    logger.info(f"Loading K-means model from {args.kmeans_model_path} ...")
    kmeans_model = joblib.load(open(args.kmeans_model_path, "rb"))
    kmeans_model.verbose = False

    _, fnames, _ = get_audio_files(args.manifest_path)

    os.makedirs(os.path.dirname(args.out_quantized_file_path), exist_ok=True)
    print(f"Writing quantized predictions to {args.out_quantized_file_path}")
    with open(args.out_quantized_file_path, "w") as fout:
        for i, feats in enumerate(features_batch):
            pred = kmeans_model.predict(feats)
            pred_str = " ".join(str(p) for p in pred)
            base_fname = os.path.basename(fnames[i]).rstrip('.'+args.extension.lstrip('.'))
            if args.channel_id is not None:
                base_fname = base_fname+f'-channel{args.channel_id}'
            if not args.hide_fname:
                fout.write(f"{base_fname}|{pred_str}\n")
            else:
                fout.write(f"{pred_str}\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
