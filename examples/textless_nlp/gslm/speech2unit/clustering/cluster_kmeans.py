# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib
from examples.textless_nlp.gslm.speech2unit.pretrained.utils import (
    get_and_dump_features,
    get_features,
)


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Learn K-means clustering over acoustic features."
    )

    # Features arguments
    parser.add_argument(
        "--in_features_path", type=str, default=None, help="Features file path"
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

    # K-means arguments
    parser.add_argument(
        "--num_clusters", type=int, help="Nubmer of clusters", default=50
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Maximum number of iterations for K-means training",
        default=150,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for K-means training",
        default=10000,
    )
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.5, type=float)
    parser.add_argument(
        "--out_kmeans_model_path",
        type=str,
        required=True,
        help="Path to save K-means model",
    )

    # Leftovers
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to use for K-means training",
        default=1369,
    )

    return parser


def get_kmeans_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
    random_state,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        tol=tol,
        max_no_improvement=max_no_improvement,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        random_state=random_state,
        verbose=1,
        compute_labels=True,
        init_size=None,
    )


def train_kmeans(kmeans_model, features_batch):
    start_time = time.time()
    kmeans_model.fit(features_batch)
    time_taken = round((time.time() - start_time) // 60, 2)
    return kmeans_model, time_taken


def main(args, logger):
    # Features loading/extraction for K-means
    if args.in_features_path:
        # Feature loading
        logger.info(f"Loading features from {args.in_features_path}...")
        features_batch = np.load(args.in_features_path, allow_pickle=True)
    else:
        # Feature extraction
        logger.info(f"Extracting {args.feature_type} acoustic features...")
        features_batch = (
            get_features(
                feature_type=args.feature_type,
                checkpoint_path=args.checkpoint_path,
                layer=args.layer,
                manifest_path=args.manifest_path,
                sample_pct=args.sample_pct,
                flatten=True,
            )
            if not args.out_features_path
            else get_and_dump_features(
                feature_type=args.feature_type,
                checkpoint_path=args.checkpoint_path,
                layer=args.layer,
                manifest_path=args.manifest_path,
                sample_pct=args.sample_pct,
                flatten=True,
                out_features_path=args.out_features_path,
            )
        )
        if args.out_features_path:
            logger.info(
                f"Saved extracted features at {args.out_features_path}"
            )
    logger.info(f"Features shape = {features_batch.shape}\n")

    # Learn and save K-means model
    kmeans_model = get_kmeans_model(
        n_clusters=args.num_clusters,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        tol=args.tol,
        max_no_improvement=args.max_no_improvement,
        n_init=args.n_init,
        reassignment_ratio=args.reassignment_ratio,
        random_state=args.seed,
    )
    logger.info("Starting k-means training...")
    kmeans_model, time_taken = train_kmeans(
        kmeans_model=kmeans_model, features_batch=features_batch
    )
    logger.info(f"...done k-means training in {time_taken} minutes")
    inertia = -kmeans_model.score(features_batch) / len(features_batch)
    logger.info(f"Total intertia: {round(inertia, 2)}\n")

    logger.info(f"Saving k-means model to {args.out_kmeans_model_path}")
    os.makedirs(os.path.dirname(args.out_kmeans_model_path), exist_ok=True)
    joblib.dump(kmeans_model, open(args.out_kmeans_model_path, "wb"))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
