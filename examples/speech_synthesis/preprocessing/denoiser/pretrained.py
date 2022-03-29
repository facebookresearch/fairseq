# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import logging

import torch.hub

from .demucs import Demucs
from .utils import deserialize_model

logger = logging.getLogger(__name__)
ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
DNS_48_URL = ROOT + "dns48-11decc9d8e3f0998.th"
DNS_64_URL = ROOT + "dns64-a7761ff99a7d5bb6.th"
MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th"


def _demucs(pretrained, url, **kwargs):
    model = Demucs(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


def dns48(pretrained=True):
    return _demucs(pretrained, DNS_48_URL, hidden=48)


def dns64(pretrained=True):
    return _demucs(pretrained, DNS_64_URL, hidden=64)


def master64(pretrained=True):
    return _demucs(pretrained, MASTER_64_URL, hidden=64)


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-m", "--model_path", help="Path to local trained model."
    )
    group.add_argument(
        "--dns48", action="store_true",
        help="Use pre-trained real time H=48 model trained on DNS."
    )
    group.add_argument(
        "--dns64", action="store_true",
        help="Use pre-trained real time H=64 model trained on DNS."
    )
    group.add_argument(
        "--master64", action="store_true",
        help="Use pre-trained real time H=64 model trained on DNS and Valentini."
    )


def get_model(args):
    """
    Load local model package or torchhub pre-trained model.
    """
    if args.model_path:
        logger.info("Loading model from %s", args.model_path)
        pkg = torch.load(args.model_path)
        model = deserialize_model(pkg)
    elif args.dns64:
        logger.info("Loading pre-trained real time H=64 model trained on DNS.")
        model = dns64()
    elif args.master64:
        logger.info(
            "Loading pre-trained real time H=64 model trained on DNS and Valentini."
        )
        model = master64()
    else:
        logger.info("Loading pre-trained real time H=48 model trained on DNS.")
        model = dns48()
    logger.debug(model)
    return model
