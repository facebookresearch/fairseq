# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import pickle

from mmpt.utils import ShardedTensor


class Shard(object):
    def __init__(
        self,
        vfeat_dir,
        tfeat_dir,
        target_dir,
        file_paths,
        shard_size=4096
    ):
        self.vfeat_dir = vfeat_dir
        self.tfeat_dir = tfeat_dir
        self.target_dir = target_dir
        self.video_ids = {}
        for split, file_path in zip(["train", "val"], file_paths):
            with open(file_path) as fr:
                self.video_ids[split] = [
                    line.strip() for line in fr.readlines()]
        self.shard_size = shard_size

    def __call__(self, split="train"):
        for split in ["train", "val"]:
            meta = {}
            for shard_idx, shard_offset in enumerate(
                range(0, len(self.video_ids[split]), self.shard_size)
            ):
                print(shard_idx)
                meta_shard = []
                video_shard = []
                for video_id in self.video_ids[split][shard_offset:shard_offset+self.shard_size]:
                    meta_shard.append(video_id)
                    npy_file = os.path.join(self.vfeat_dir, video_id + ".npy")
                    video_shard.append(np.load(npy_file))

                meta[shard_idx] = meta_shard
                video_shard = ShardedTensor.from_list(video_shard)
                target_path = os.path.join(
                    self.target_dir, split + "_" + str(shard_idx))
                video_shard.save(target_path)

            target_path = os.path.join(self.target_dir, split + "_meta")
            with open(target_path + ".pkl", "wb") as fw:
                pickle.dump(meta, fw, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    shard = Shard(
        "data/feat/feat_how2_s3d",
        "data/how2/raw_caption_dedup.bert-base-uncased",
        "data/feat/feat_how2_s3d_shard_small",
        ["data/how2/how2_s3d_train.lst", "data/how2/how2_s3d_val.lst"]
    )

    shard()
