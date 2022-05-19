# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import os
import argparse
import numpy as np

from torch.utils.data import Dataset, DataLoader
from mmpt.processors import PKLJSONStrTextProcessor
from mmpt.utils import ShardedTensor, recursive_config


class TokenizerDataset(Dataset):
    def __init__(self, config):
        self.text_processor = PKLJSONStrTextProcessor(config)
        self.video_ids = list(self.text_processor.data.keys())

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        return video_id, self.text_processor(video_id)

    def __len__(self):
        return len(self.video_ids)


def numpify(shard_idx, video_ids, captions, target_dir, split, prefix, max_cap_len=32):
    startends = []
    caps_ids = []
    for video_id in video_ids:
        caption = captions[video_id]
        startend = []
        cap_ids = []
        for start, end, cap in zip(
                caption["start"], caption["end"], caption["cap"]):
            startend.append(np.array([start, end]).astype("float32"))
            cap_id = np.full((max_cap_len,), -1, dtype=np.int32)
            cap = cap[:max_cap_len]
            cap_id[:len(cap)] = cap
            cap_ids.append(cap_id)
        startends.append(np.stack(startend))
        caps_ids.append(np.stack(cap_ids))

    startends = ShardedTensor.from_list(startends)
    target_path = os.path.join(
        target_dir,
        prefix + split + "_" + str(shard_idx)
    )
    print("save to", target_path)
    startends.save(target_path + ".startends")
    caps_ids = ShardedTensor.from_list(caps_ids)
    caps_ids.save(target_path + ".caps_ids")


def sharding(config, out_file):
    with open(out_file, "rb") as fr:
        captions = pickle.load(fr)
    target_dir = config.target_dir
    prefix = os.path.basename(
                os.path.splitext(config.caption_pkl_path)[0]
            ) + "." + config.bert_name + "."
    for split in ["train", "val"]:
        target_path = os.path.join(target_dir, split + "_meta")
        with open(target_path + ".pkl", "rb") as fr:
            meta = pickle.load(fr)
        print("load meta", target_path, len(meta))
        for shard_id in meta:
            numpify(
                shard_id, meta[shard_id], captions,
                target_dir, split, prefix
            )


def tokenize(config, out_file):
    def collator(samples):
        return samples
    dataset = TokenizerDataset(config)
    data = {}
    for idx, batch in enumerate(
            DataLoader(dataset, collate_fn=collator, num_workers=16)):
        for video_id, caption in batch:
            data[video_id] = caption
        if idx % 5000 == 0:
            print(idx)
    with open(out_file, "wb") as fw:
        pickle.dump(data, fw, pickle.HIGHEST_PROTOCOL)


def main(args):
    config = recursive_config(args.config).dataset

    out_file = os.path.splitext(config.caption_pkl_path)[0] \
        + "." + config.bert_name + ".pkl"
    if not os.path.isfile(out_file):
        tokenize(config, out_file)
    sharding(config, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretokenize (raw_)caption.json into pkl.")
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    main(args)
