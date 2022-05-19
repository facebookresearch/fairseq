# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import urllib.parse
import json
import pandas as pd

from tqdm import tqdm


# TODO: extending to other datasets.
supported_formats = {}


class PathBuilder(object):
    @classmethod
    def build(cls, video_dirs, feature_dir, ext, shards=0, split=None):
        meta_fn = os.path.join(feature_dir, "meta_plan.json")
        os.makedirs(feature_dir, exist_ok=True)
        if os.path.isfile(meta_fn):
            with open(meta_fn) as fr:
                meta = json.load(fr)
                return meta
        print("searching videos...")

        video_id_to_path = {}
        for video_dir in video_dirs.split(","):
            # TODO: add supports of recursive listdir.
            if video_dir in supported_formats:
                supported_formats[video_dir].load(video_dir, video_id_to_path)
            else:
                for idx, fn in enumerate(tqdm(os.listdir(video_dir))):
                    video_fn = os.path.join(video_dir, fn)
                    if os.path.isfile(video_fn):
                        video_id = os.path.splitext(fn)[0]
                        video_id_to_path[video_id] = video_fn
                    elif os.path.isdir(video_fn):
                        # shards of folders.
                        shard_dir = video_fn
                        for idx, fn in enumerate(os.listdir(shard_dir)):
                            video_fn = os.path.join(shard_dir, fn)
                            if os.path.isfile(video_fn):
                                video_id = os.path.splitext(fn)[0]
                                video_id_to_path[video_id] = video_fn

        video_path, feature_path = [], []
        valid_ext = set()
        for idx, video_id in enumerate(video_id_to_path):
            video_path.append(video_id_to_path[video_id])
            if ext is None:
                # use original file ext for format compatibility.
                video_id_to_path[video_id]
                path = urllib.parse.urlparse(video_id_to_path[video_id]).path
                ext = os.path.splitext(path)[1]
            if ext not in valid_ext:
                valid_ext.add(ext)
                print("adding", ext)
            if shards:
                shard_id = str(idx % shards)
                feature_fn = os.path.join(
                    feature_dir, shard_id, video_id + ext)
            else:
                feature_fn = os.path.join(
                    feature_dir, video_id + ext)
            feature_path.append(feature_fn)

        print("targeting", len(feature_path), "videos")
        meta = {
            "video_path": video_path, "feature_path": feature_path}
        with open(meta_fn, "w") as fw:
            json.dump(meta, fw)

        if split is not None:
            splits = split.split("/")
            assert len(splits) == 2
            cur, total = int(splits[0]), int(splits[1])
            assert cur < total
            import math
            chunk = math.ceil(len(meta["video_path"]) / total)
            start = cur * chunk
            end = (cur + 1) * chunk
            meta = {
                    "video_path": meta["video_path"][start:end],
                    "feature_path": meta["feature_path"][start:end]
            }

        return meta
