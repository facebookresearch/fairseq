# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .how2processor import (
    ShardedHow2MetaProcessor,
    ShardedVideoProcessor,
    ShardedTextProcessor,
    VariedLenAligner,
    OverlappedAligner
)


class ShardedHow2VideoRetriMetaProcessor(ShardedHow2MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.num_video_per_batch = config.num_video_per_batch
        self.cands = [
            self.data[batch_offset:batch_offset + self.num_video_per_batch]
            for batch_offset in
            range(0, (len(self.data) // (8 * self.num_video_per_batch)) * 8 * self.num_video_per_batch, self.num_video_per_batch)]

    def __len__(self):
        return len(self.cands)

    def set_candidates(self, cands):
        # no changes on num of batches.
        print(len(self.cands), "->", len(cands))
        # assert len(self.cands) == len(cands)
        self.cands = cands

    def __getitem__(self, idx):
        video_ids = self.cands[idx]
        assert isinstance(video_ids, list)
        sharded_video_idxs = []
        for video_id in video_ids:
            shard_id, video_idx = self.video_id_to_shard[video_id]
            sharded_video_idxs.append((video_id, -1, shard_id, video_idx))
        return sharded_video_idxs, sharded_video_idxs


class ShardedVideoRetriVideoProcessor(ShardedVideoProcessor):
    """In retrival case the video_id
    is a list of tuples: `(shard_id, video_idx)` ."""

    def __call__(self, sharded_video_idxs):
        assert isinstance(sharded_video_idxs, list)
        cand_feats = []
        for shared_video_idx in sharded_video_idxs:
            feat = super().__call__(shared_video_idx)
            cand_feats.append(feat)
        return cand_feats


class ShardedVideoRetriTextProcessor(ShardedTextProcessor):
    """In retrival case the video_id
    is a list of tuples: `(shard_id, video_idx)` ."""

    def __call__(self, sharded_video_idxs):
        assert isinstance(sharded_video_idxs, list)
        cand_caps = []
        for shared_video_idx in sharded_video_idxs:
            caps = super().__call__(shared_video_idx)
            cand_caps.append(caps)
        return cand_caps


class VideoRetriAligner(VariedLenAligner):
    # Retritask will trim dim-0.
    def __call__(self, sharded_video_idxs, video_features, text_features):
        from transformers import default_data_collator
        batch, video_ids = [], []
        for video_id, video_feature, text_feature in \
                zip(sharded_video_idxs, video_features, text_features):
            sub_batch = super().__call__(video_id, video_feature, text_feature)
            batch.append(sub_batch)
            if isinstance(video_id, tuple):
                video_id = video_id[0]
            video_ids.append(video_id)
        batch = default_data_collator(batch)
        batch["video_id"] = video_ids
        return batch


class VideoRetriOverlappedAligner(OverlappedAligner):
    # Retritask will trim dim-0.
    def __call__(self, sharded_video_idxs, video_features, text_features):
        from transformers import default_data_collator
        batch, video_ids = [], []
        for video_id, video_feature, text_feature in \
                zip(sharded_video_idxs, video_features, text_features):
            sub_batch = super().__call__(video_id, video_feature, text_feature)
            batch.append(sub_batch)
            if isinstance(video_id, tuple):
                video_id = video_id[0]
            video_ids.append(video_id)
        batch = default_data_collator(batch)
        batch["video_id"] = video_ids
        return batch
