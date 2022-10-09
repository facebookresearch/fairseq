#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implement unsupervised metric for decoding hyperparameter selection:
    $$ alpha * LM_PPL + ViterbitUER(%) * 100 $$
"""
import argparse
import logging
import sys

import editdistance

logging.root.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--hypo", help="hypo transcription", required=True)
    parser.add_argument(
        "-r", "--reference", help="reference transcription", required=True
    )
    return parser


def compute_wer(ref_uid_to_tra, hyp_uid_to_tra, g2p):
    d_cnt = 0
    w_cnt = 0
    w_cnt_h = 0
    for uid in hyp_uid_to_tra:
        ref = ref_uid_to_tra[uid].split()
        if g2p is not None:
            hyp = g2p(hyp_uid_to_tra[uid])
            hyp = [p for p in hyp if p != "'" and p != " "]
            hyp = [p[:-1] if p[-1].isnumeric() else p for p in hyp]
        else:
            hyp = hyp_uid_to_tra[uid].split()
        d_cnt += editdistance.eval(ref, hyp)
        w_cnt += len(ref)
        w_cnt_h += len(hyp)
    wer = float(d_cnt) / w_cnt
    logger.debug(
        (
            f"wer = {wer * 100:.2f}%; num. of ref words = {w_cnt}; "
            f"num. of hyp words = {w_cnt_h}; num. of sentences = {len(ref_uid_to_tra)}"
        )
    )
    return wer


def main():
    args = get_parser().parse_args()

    errs = 0
    count = 0
    with open(args.hypo, "r") as hf, open(args.reference, "r") as rf:
        for h, r in zip(hf, rf):
            h = h.rstrip().split()
            r = r.rstrip().split()
            errs += editdistance.eval(r, h)
            count += len(r)

    logger.info(f"UER: {errs / count * 100:.2f}%")


if __name__ == "__main__":
    main()


def load_tra(tra_path):
    with open(tra_path, "r") as f:
        uid_to_tra = {}
        for line in f:
            uid, tra = line.split(None, 1)
            uid_to_tra[uid] = tra
    logger.debug(f"loaded {len(uid_to_tra)} utterances from {tra_path}")
    return uid_to_tra
