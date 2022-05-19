# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import pickle
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..processors import (
    ShardedHow2MetaProcessor,
    ShardedVideoProcessor,
    ShardedTextProcessor,
    VariedLenAligner,
)

from ..datasets import MMDataset
from .task import Task
from ..modules import vectorpool
from ..evaluators.predictor import Predictor
from ..utils import set_seed, get_local_rank, get_world_size


class RetriTask(Task):
    """abstract class for task with retrival."""

    def reshape_subsample(self, sample):
        for key in sample:
            if torch.is_tensor(sample[key]):
                sample[key] = self.flat_subsample(sample[key])
        return sample

    def flat_subsample(self, tensor):
        if tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        return tensor

    def build_dataloader(self):
        """called by `get_batch_iterator` in fairseqmmtask. """
        # TODO: hard-code dataloader for retri for now and configurable in .yaml.
        # reuse the `train.lst`.
        self.config.dataset.split = "train"
        meta_processor = ShardedHow2MetaProcessor(self.config.dataset)
        video_processor = ShardedVideoProcessor(self.config.dataset)
        text_processor = ShardedTextProcessor(self.config.dataset)

        aligner = VariedLenAligner(self.config.dataset)
        aligner.subsampling = self.config.dataset.clip_per_video

        self.retri_data = MMDataset(
            meta_processor, video_processor, text_processor, aligner
        )

        retri_sampler = DistributedSampler(self.retri_data)
        infer_scale = 16
        batch_size = self.config.dataset.num_video_per_batch \
            * infer_scale

        self.retri_dataloader = DataLoader(
            self.retri_data,
            collate_fn=self.retri_data.collater,
            batch_size=batch_size,
            shuffle=False,
            sampler=retri_sampler,
            num_workers=self.config.fairseq.dataset.num_workers
        )
        return self.retri_dataloader

    def retrive_candidates(self, epoch, dataloader=None):
        if get_local_rank() == 0:
            print("running retrieval model.")
        out_dir = os.path.join(
            self.config.fairseq.checkpoint.save_dir, "retri")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isfile(
                os.path.join(
                    out_dir, "batched_e" + str(epoch) + "_videos0.pkl")
        ):
            if dataloader is None:
                dataloader = self.retri_dataloader

            self.model.eval()
            self.model.is_train = False

            assert self.retri_data.meta_processor.data == \
                self.train_data.meta_processor.data  # video_ids not mutated.

            self._retri_predict(epoch, dataloader)

            self.model.train()
            self.model.is_train = True

        torch.distributed.barrier()
        output = self._retri_sync(epoch, out_dir)
        torch.distributed.barrier()
        self.train_data.meta_processor.set_candidates(output)
        return output


class VideoRetriTask(RetriTask):
    """RetriTask on video level."""

    def reshape_subsample(self, sample):
        if (
            hasattr(self.config.dataset, "clip_per_video")
            and self.config.dataset.clip_per_video is not None
            and self.config.dataset.clip_per_video > 1
        ):
            for key in sample:
                if torch.is_tensor(sample[key]):
                    sample[key] = self.flat_subsample(sample[key])
        return sample

    def flat_subsample(self, tensor):
        if tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        return Task.flat_subsample(self, tensor)

    def _retri_predict(self, epoch, dataloader):
        set_seed(epoch)
        # save for retrival.
        predictor = VideoPredictor(self.config)
        predictor.predict_loop(
            self.model, dataloader)
        set_seed(epoch)  # get the same text clips.
        # retrival.
        retri_predictor = VideoRetriPredictor(
            self.config)
        retri_predictor.predict_loop(
            self.model, predictor.vecpool.retriver, epoch)
        del predictor
        del retri_predictor

    def _retri_sync(self, epoch, out_dir):
        # gpu do the same merge.
        batched_videos = []
        for local_rank in range(get_world_size()):
            fn = os.path.join(
                out_dir,
                "batched_e" + str(epoch) + "_videos" + str(local_rank) + ".pkl")
            with open(fn, "rb") as fr:
                batched_videos.extend(pickle.load(fr))
        print(
            "[INFO] batched_videos",
            len(batched_videos), len(batched_videos[0]))
        return batched_videos


class VideoPredictor(Predictor):
    def __init__(self, config):
        vectorpool_cls = getattr(vectorpool, config.vectorpool_cls)
        self.vecpool = vectorpool_cls(config)

    def predict_loop(
        self,
        model,
        dataloader,
        early_stop=-1,
    ):
        with torch.no_grad():
            if get_local_rank() == 0:
                dataloader = tqdm(dataloader)
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx == early_stop:
                    break
                self(batch, model)
        return self.finalize()

    def __call__(self, sample, model, **kwargs):
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
        subsample = sample["vfeats"].size(1)
        sample = self.to_ctx(sample, device, dtype)
        for key in sample:
            if torch.is_tensor(sample[key]):
                size = sample[key].size()
                if len(size) >= 2:
                    batch_size = size[0] * size[1]
                    expanded_size = (
                        (batch_size,) + size[2:] if len(size) > 2
                        else (batch_size,)
                    )
                    sample[key] = sample[key].view(expanded_size)

        outputs = model(**sample)
        sample.update(outputs)
        self.vecpool(sample, subsample)

    def finalize(self):
        print("[INFO]", self.vecpool)
        if not self.vecpool.retriver.db.is_trained:
            self.vecpool.retriver.finalize_training()
        return self.vecpool.retriver


class VideoRetriPredictor(Predictor):
    """
    Online Retrieval Predictor for Clips (used by RetriTask).
    TODO: merge this with VisPredictor?
    """

    def __init__(self, config):
        self.pred_dir = os.path.join(
            config.fairseq.checkpoint.save_dir,
            "retri")
        self.num_cands = config.num_cands
        self.num_video_per_batch = config.dataset.num_video_per_batch

    def predict_loop(
        self,
        model,
        retriver,
        epoch,
        early_stop=-1
    ):
        # a fake loop that only try to recover video vector
        # from video_id.
        batched_videos = []
        # obtain available video_ids.
        video_ids = list(retriver.videoid_to_vectoridx.keys())

        dataloader = random.sample(
            video_ids,
            len(video_ids) // self.num_video_per_batch
        )

        if get_local_rank() == 0:
            dataloader = tqdm(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            # batch is one video id.
            if batch_idx == early_stop:
                break
            video_ids = retriver.search_by_video_ids(
                [batch], self.num_cands)[0]
            if len(video_ids) > self.num_video_per_batch:
                # we moved the center to make cluster robust.
                video_ids = random.sample(video_ids, self.num_video_per_batch)
            batched_videos.append(video_ids)
        return self.finalize(batched_videos, epoch)

    def finalize(self, batched_videos, epoch):
        fn = os.path.join(
            self.pred_dir,
            "batched_e" + str(epoch) + "_videos" + str(get_local_rank()) + ".pkl")
        with open(fn, "wb") as fw:
            pickle.dump(batched_videos, fw, pickle.HIGHEST_PROTOCOL)
        return batched_videos
