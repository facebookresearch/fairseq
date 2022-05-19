# Copyright (c) Facebook, Inc. All Rights Reserved

import torch
import os
import numpy as np
import pickle

from . import retri
from ..utils import get_local_rank


class VectorPool(object):
    """
    Base class of retrieval space.
    """

    def __init__(self, config):
        from transformers import AutoConfig
        self.hidden_size = AutoConfig.from_pretrained(
            config.dataset.bert_name).hidden_size
        self.retriever_cls = getattr(retri, config.retriever_cls)

    def __call__(self, sample, **kwargs):
        raise NotImplementedError

    def build_retriver(
        self,
        retriever_cls=None,
        hidden_size=None,
        centroids=512,
        db_type="flatl2",
        examples_per_cent_to_train=48
    ):

        """merge results from multiple gpus and return a retriver.."""
        self.retriver = retriever_cls(
            hidden_size, centroids, db_type, examples_per_cent_to_train)
        return self.retriver

    def __repr__(self):
        if hasattr(self, "retriver"):
            retriver_name = str(len(self.retriver))
        else:
            retriver_name = "no retriver field yet"
        return self.__class__.__name__ \
            + "(" + retriver_name + ")"


class VideoVectorPool(VectorPool):
    """
    average clips of a video as video representation.
    """
    def __init__(self, config):
        super().__init__(config)
        self.build_retriver(self.retriever_cls, self.hidden_size)

    def __call__(self, sample, subsampling, **kwargs):
        hidden_states = (
            sample["pooled_video"] + sample["pooled_text"]) / 2.
        hidden_states = hidden_states.view(
            -1, subsampling,
            hidden_states.size(-1))
        hidden_states = torch.mean(hidden_states, dim=1)
        hidden_states = hidden_states.cpu().detach().numpy()
        video_ids = []
        for offset_idx, video_id in enumerate(sample["video_id"]):
            if isinstance(video_id, tuple) and len(video_id) == 3:
                # a sharded video_id.
                video_id = video_id[0]
            video_ids.append(video_id)
        assert len(video_ids) == len(hidden_states)
        self.retriver.add(
            hidden_states.astype("float32"),
            video_ids
        )


class DistributedVectorPool(VectorPool):
    """
    support sync of multiple gpus/nodes.
    """
    def __init__(self, config):
        super().__init__(config)
        self.out_dir = os.path.join(
            config.fairseq.checkpoint.save_dir,
            "retri")
        os.makedirs(self.out_dir, exist_ok=True)
        self.hidden_states = []
        self.video_ids = []

    def build_retriver(
        self,
        retriever_cls=None,
        hidden_size=None,
        centroids=4096,
        db_type="flatl2",
        examples_per_cent_to_train=48
    ):
        if retriever_cls is None:
            retriever_cls = self.retriever_cls
        if hidden_size is None:
            hidden_size = self.hidden_size
        """merge results from multiple gpus and return a retriver.."""
        if torch.distributed.is_initialized():
            self.save()
            # sync saving.
            torch.distributed.barrier()
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        self.retriver = retriever_cls(
            hidden_size, centroids, db_type, examples_per_cent_to_train)
        # each gpu process has its own retriever.
        for local_rank in range(world_size):
            if get_local_rank() == 0:
                print("load local_rank", local_rank)
            hidden_states, video_ids = self.load(local_rank)
            hidden_states = hidden_states.astype("float32")
            self.retriver.add(hidden_states, video_ids)
        return self.retriver

    def load(self, local_rank):
        hidden_states = np.load(
            os.path.join(
                self.out_dir,
                "hidden_state" + str(local_rank) + ".npy"
            )
        )

        with open(
            os.path.join(
                self.out_dir, "video_id" + str(local_rank) + ".pkl"),
                "rb") as fr:
            video_ids = pickle.load(fr)
        return hidden_states, video_ids

    def save(self):
        hidden_states = np.vstack(self.hidden_states)
        assert len(hidden_states) == len(self.video_ids), "{}, {}".format(
            len(hidden_states),
            len(self.video_ids)
        )
        local_rank = torch.distributed.get_rank() \
            if torch.distributed.is_initialized() else 0

        np.save(
            os.path.join(
                self.out_dir,
                "hidden_state" + str(local_rank) + ".npy"),
            hidden_states)

        with open(
            os.path.join(
                self.out_dir,
                "video_id" + str(local_rank) + ".pkl"),
                "wb") as fw:
            pickle.dump(
                self.video_ids,
                fw,
                protocol=pickle.HIGHEST_PROTOCOL
            )


class DistributedVideoVectorPool(DistributedVectorPool):
    """
    average clips of a video as video representation.
    """
    def __call__(self, sample, subsampling, **kwargs):
        hidden_states = (
            sample["pooled_video"] + sample["pooled_text"]) / 2.
        hidden_states = hidden_states.view(
            -1, subsampling,
            hidden_states.size(-1))
        hidden_states = torch.mean(hidden_states, dim=1)
        hidden_states = hidden_states.cpu().detach().numpy()
        video_ids = []
        for offset_idx, video_id in enumerate(sample["video_id"]):
            if isinstance(video_id, tuple) and len(video_id) == 3:
                # a sharded video_id.
                video_id = video_id[0]
            video_ids.append(video_id)
        assert len(video_ids) == len(hidden_states)
        self.hidden_states.append(hidden_states)
        self.video_ids.extend(video_ids)


# ------------ the following are deprecated --------------

class TextClipVectorPool(VectorPool):
    def __init__(self, config):
        from transformers import AutoConfig
        hidden_size = AutoConfig.from_pretrained(
            config.dataset.bert_name).hidden_size
        retriever_cls = getattr(retri, config.retriever_cls)
        self.build_retriver(retriever_cls, hidden_size)

    def __call__(self, sample, **kwargs):
        clip_meta = sample["clip_meta"].cpu()
        assert torch.all(torch.le(clip_meta[:, 4], clip_meta[:, 5]))
        text_meta = [tuple(item.tolist()) for item in clip_meta[:, 3:]]

        if hasattr(self, "retriver"):
            # build_retriver is called.
            self.retriver.add(
                sample["pooled_text"].cpu().numpy().astype("float32"),
                text_meta
            )
        else:
            raise NotImplementedError


class MMClipVectorPool(VectorPool):
    """
    Multimodal Clip-level vector pool.
    """
    def __init__(self, out_dir):
        """use hidden_states to store `(video, text)`."""
        """use video_ids to store `(video_id, start, end)`."""
        super().__init__(out_dir)

    def __call__(self, sample, **kwargs):
        pooled_video = sample["pooled_video"].cpu().unsqueeze(1).numpy()
        pooled_text = sample["pooled_text"].cpu().unsqueeze(1).numpy()

        self.hidden_states.append(
            np.concatenate([pooled_video, pooled_text], axis=1)
        )

        video_starts = sample["video_start"].cpu()
        video_ends = sample["video_end"].cpu()
        assert torch.all(torch.le(video_starts, video_ends))

        text_starts = sample["text_start"].cpu()
        text_ends = sample["text_end"].cpu()
        assert torch.all(torch.le(text_starts, text_ends))
        subsample_size = sample["pooled_video"].size(0) // len(sample["video_id"])
        video_ids = [video_id for video_id in sample["video_id"]
                    for _ in range(subsample_size)
        ]
        for video_id, video_start, video_end, text_start, text_end in zip(
                video_ids, video_starts, video_ends, text_starts, text_ends):
            self.video_ids.append((
                video_id,
                (int(video_start), int(video_end)),
                (int(text_start), int(text_end))
            ))
