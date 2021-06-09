# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import math
import os
import os.path as op
import sys

import tqdm
from dump_hubert_feature import HubertFeatureReader
from fairseq.data.audio.audio_utils import get_waveform
from fairseq.data.audio.speech_to_text_dataset import (
    read_from_uncompressed_zip,
)
from npy_append_array import NpyAppendArray

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature_s2t")


class HubertFeatureReaderS2T(HubertFeatureReader):
    def read_audio(self, path, ref_len=None):
        path, *extra = path.split(":")
        assert len(extra) == 2
        assert path.endswith(".zip")

        data = read_from_uncompressed_zip(path, int(extra[0]), int(extra[1]))
        f = io.BytesIO(data)
        wav, sr = get_waveform(f)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav


def get_path_iterator(root, tsv, nshard, rank):
    with open(tsv) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        subpaths = [op.join(root, e["audio"]) for e in reader]

        tot = len(subpaths)
        shard_size = math.ceil(tot / nshard)
        start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
        assert start < end, "start={start}, end={end}"
        logger.info(
            f"rank {rank} of {nshard}, process {end-start} "
            f"({start}-{end}) out of {tot}"
        )

        subpaths = subpaths[start:end]

        def iterate():
            for subpath in subpaths:
                yield op.join(root, subpath)

        return iterate, len(subpaths)


def dump_feature(
    root,
    tsv_path,
    ckpt_path,
    layer,
    nshard,
    rank,
    feat_dir,
    feat_name,
    max_chunk,
):
    reader = HubertFeatureReaderS2T(ckpt_path, layer, max_chunk)
    generator, num = get_path_iterator(root, tsv_path, nshard, rank)
    iterator = generator()

    feat_path = f"{feat_dir}/{feat_name}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{feat_name}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if op.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path)
            feat_f.append(feat.cpu().numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("tsv_path")
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("feat_name")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    dump_feature(**vars(args))
