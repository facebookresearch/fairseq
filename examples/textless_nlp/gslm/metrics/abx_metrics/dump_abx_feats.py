# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import joblib
import numpy as np

from examples.textless_nlp.gslm.speech2unit.clustering.utils import get_audio_files
from examples.textless_nlp.gslm.speech2unit.pretrained.utils import get_features

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
        "--kmeans_model_path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Pretrained model checkpoint",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--out_dir_path",
        required=True,
        type=str,
        help="File path of quantized output.",
    )
    parser.add_argument(
        "--extension", type=str, default=".flac", help="Features file path"
    )
    return parser


def one_hot(feat, n_clusters):
    return np.eye(n_clusters)[feat]

def main(args, logger):
    # Feature extraction
    logger.info(f"Extracting {args.feature_type} acoustic features...")
    features_batch = get_features(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint_path,
        layer=args.layer,
        manifest_path=args.manifest_path,
        sample_pct=1.0,
        flatten=False,
    )
    logger.info(f"Features extracted for {len(features_batch)} utterances.\n")
    logger.info(f"Dimensionality of representation = {features_batch[0].shape[1]}")

    logger.info(f"Loading K-means model from {args.kmeans_model_path} ...")
    kmeans_model = joblib.load(open(args.kmeans_model_path, "rb"))
    kmeans_model.verbose = False

    _, fnames, _ = get_audio_files(args.manifest_path)

    os.makedirs(args.out_dir_path, exist_ok=True)
    logger.info(f"Writing quantized features to {args.out_dir_path}")
    for i, feats in enumerate(features_batch):
        pred = kmeans_model.predict(feats)
        emb = one_hot(pred, kmeans_model.n_clusters)
        base_fname = os.path.basename(fnames[i]).rstrip(args.extension)
        output_path = os.path.join(args.out_dir_path, f"{base_fname}.npy")
        with open(output_path, "wb") as f:
            np.save(f, emb)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
