# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import sys

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
import tqdm
from npy_append_array import NpyAppendArray

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
            return torch.cat(feat, 1).squeeze(0)


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        tot = len(lines)
        shard_size = math.ceil(tot / nshard)
        start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
        assert start < end, "start={start}, end={end}"
        logger.info(
            f"rank {rank} of {nshard}, process {end-start} "
            f"({start}-{end}) out of {tot}"
        )

        lines = lines[start:end]

        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)

        return iterate, len(lines)


def dump_feature(
    tsv_dir, split, ckpt_path, layer, nshard, rank, feat_dir, max_chunk
):
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    iterator = generator()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path, nsample)
            feat_f.append(feat.cpu().numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    dump_feature(**vars(args))
