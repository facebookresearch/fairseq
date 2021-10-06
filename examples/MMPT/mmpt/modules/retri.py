# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import pickle
import time

try:
    import faiss
except ImportError:
    pass

from collections import defaultdict

from ..utils import get_local_rank, print_on_rank0


class VectorRetriever(object):
    """
    How2 Video Retriver.
    Reference usage of FAISS:
    https://github.com/fairinternal/fairseq-py/blob/paraphrase_pretraining/fairseq/data/multilingual_faiss_dataset.py
    """

    def __init__(self, hidden_size, cent, db_type, examples_per_cent_to_train):
        if db_type == "flatl2":
            quantizer = faiss.IndexFlatL2(hidden_size)  # the other index
            self.db = faiss.IndexIVFFlat(
                quantizer, hidden_size, cent, faiss.METRIC_L2)
        elif db_type == "pq":
            self.db = faiss.index_factory(
                    hidden_size, f"IVF{cent}_HNSW32,PQ32"
            )
        else:
            raise ValueError("unknown type of db", db_type)
        self.train_thres = cent * examples_per_cent_to_train
        self.train_cache = []
        self.train_len = 0
        self.videoid_to_vectoridx = {}
        self.vectoridx_to_videoid = None
        self.make_direct_maps_done = False

    def make_direct_maps(self):
        faiss.downcast_index(self.db).make_direct_map()

    def __len__(self):
        return self.db.ntotal

    def save(self, out_dir):
        faiss.write_index(
            self.db,
            os.path.join(out_dir, "faiss_idx")
        )
        with open(
                os.path.join(
                    out_dir, "videoid_to_vectoridx.pkl"),
                "wb") as fw:
            pickle.dump(
                self.videoid_to_vectoridx, fw,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def load(self, out_dir):
        fn = os.path.join(out_dir, "faiss_idx")
        self.db = faiss.read_index(fn)
        with open(
                os.path.join(out_dir, "videoid_to_vectoridx.pkl"), "rb") as fr:
            self.videoid_to_vectoridx = pickle.load(fr)

    def add(self, hidden_states, video_ids, last=False):
        assert len(hidden_states) == len(video_ids), "{}, {}".format(
            str(len(hidden_states)), str(len(video_ids)))
        assert len(hidden_states.shape) == 2
        assert hidden_states.dtype == np.float32

        valid_idx = []
        for idx, video_id in enumerate(video_ids):
            if video_id not in self.videoid_to_vectoridx:
                valid_idx.append(idx)
                self.videoid_to_vectoridx[video_id] = \
                    len(self.videoid_to_vectoridx)

        hidden_states = hidden_states[valid_idx]
        if not self.db.is_trained:
            self.train_cache.append(hidden_states)
            self.train_len += hidden_states.shape[0]
            if self.train_len < self.train_thres:
                return
            self.finalize_training()
        else:
            self.db.add(hidden_states)

    def finalize_training(self):
        hidden_states = np.concatenate(self.train_cache, axis=0)
        del self.train_cache
        local_rank = get_local_rank()
        if local_rank == 0:
            start = time.time()
            print("training db on", self.train_thres, "/", self.train_len)
        self.db.train(hidden_states[:self.train_thres])
        if local_rank == 0:
            print("training db for", time.time() - start)
        self.db.add(hidden_states)

    def search(
        self,
        query_hidden_states,
        orig_dist,
    ):
        if len(self.videoid_to_vectoridx) != self.db.ntotal:
            raise ValueError(
                "cannot search: size mismatch in-between index and db",
                len(self.videoid_to_vectoridx),
                self.db.ntotal
            )

        if self.vectoridx_to_videoid is None:
            self.vectoridx_to_videoid = {
                self.videoid_to_vectoridx[videoid]: videoid
                for videoid in self.videoid_to_vectoridx
            }
            assert len(self.vectoridx_to_videoid) \
                == len(self.videoid_to_vectoridx)

        # MultilingualFaissDataset uses the following; not sure the purpose.
        # faiss.ParameterSpace().set_index_parameter(self.db, "nprobe", 10)
        queried_dist, index = self.db.search(query_hidden_states, 1)
        queried_dist, index = queried_dist[:, 0], index[:, 0]

        outputs = np.array(
            [self.vectoridx_to_videoid[_index]
                if _index != -1 else (-1, -1, -1) for _index in index],
            dtype=np.int32)
        outputs[queried_dist <= orig_dist] = -1
        return outputs

    def search_by_video_ids(
        self,
        video_ids,
        retri_factor
    ):
        if len(self.videoid_to_vectoridx) != self.db.ntotal:
            raise ValueError(
                len(self.videoid_to_vectoridx),
                self.db.ntotal
            )

        if not self.make_direct_maps_done:
            self.make_direct_maps()

        if self.vectoridx_to_videoid is None:
            self.vectoridx_to_videoid = {
                self.videoid_to_vectoridx[videoid]: videoid
                for videoid in self.videoid_to_vectoridx
            }
            assert len(self.vectoridx_to_videoid) \
                == len(self.videoid_to_vectoridx)

        query_hidden_states = []
        vector_ids = []
        for video_id in video_ids:
            vector_id = self.videoid_to_vectoridx[video_id]
            vector_ids.append(vector_id)
            query_hidden_state = self.db.reconstruct(vector_id)
            query_hidden_states.append(query_hidden_state)
        query_hidden_states = np.stack(query_hidden_states)

        # MultilingualFaissDataset uses the following; not sure the reason.
        # faiss.ParameterSpace().set_index_parameter(self.db, "nprobe", 10)
        _, index = self.db.search(query_hidden_states, retri_factor)
        outputs = []
        for sample_idx, sample in enumerate(index):
            # the first video_id is always the video itself.
            cands = [video_ids[sample_idx]]
            for vector_idx in sample:
                if vector_idx >= 0 \
                        and vector_ids[sample_idx] != vector_idx:
                    cands.append(
                        self.vectoridx_to_videoid[vector_idx]
                    )
            outputs.append(cands)
        return outputs


class VectorRetrieverDM(VectorRetriever):
    """
    with direct map.
    How2 Video Retriver.
    Reference usage of FAISS:
    https://github.com/fairinternal/fairseq-py/blob/paraphrase_pretraining/fairseq/data/multilingual_faiss_dataset.py
    """

    def __init__(
        self,
        hidden_size,
        cent,
        db_type,
        examples_per_cent_to_train
    ):
        super().__init__(
            hidden_size, cent, db_type, examples_per_cent_to_train)
        self.make_direct_maps_done = False

    def make_direct_maps(self):
        faiss.downcast_index(self.db).make_direct_map()
        self.make_direct_maps_done = True

    def search(
        self,
        query_hidden_states,
        orig_dist,
    ):
        if len(self.videoid_to_vectoridx) != self.db.ntotal:
            raise ValueError(
                len(self.videoid_to_vectoridx),
                self.db.ntotal
            )

        if not self.make_direct_maps_done:
            self.make_direct_maps()
        if self.vectoridx_to_videoid is None:
            self.vectoridx_to_videoid = {
                self.videoid_to_vectoridx[videoid]: videoid
                for videoid in self.videoid_to_vectoridx
            }
            assert len(self.vectoridx_to_videoid) \
                == len(self.videoid_to_vectoridx)

        # MultilingualFaissDataset uses the following; not sure the reason.
        # faiss.ParameterSpace().set_index_parameter(self.db, "nprobe", 10)
        queried_dist, index = self.db.search(query_hidden_states, 1)
        outputs = []
        for sample_idx, sample in enumerate(index):
            # and queried_dist[sample_idx] < thres \
            if sample >= 0 \
                    and queried_dist[sample_idx] < orig_dist[sample_idx]:
                outputs.append(self.vectoridx_to_videoid[sample])
            else:
                outputs.append(None)
        return outputs

    def search_by_video_ids(
        self,
        video_ids,
        retri_factor=8
    ):
        if len(self.videoid_to_vectoridx) != self.db.ntotal:
            raise ValueError(
                len(self.videoid_to_vectoridx),
                self.db.ntotal
            )

        if not self.make_direct_maps_done:
            self.make_direct_maps()
        if self.vectoridx_to_videoid is None:
            self.vectoridx_to_videoid = {
                self.videoid_to_vectoridx[videoid]: videoid
                for videoid in self.videoid_to_vectoridx
            }
            assert len(self.vectoridx_to_videoid) \
                == len(self.videoid_to_vectoridx)

        query_hidden_states = []
        vector_ids = []
        for video_id in video_ids:
            vector_id = self.videoid_to_vectoridx[video_id]
            vector_ids.append(vector_id)
            query_hidden_state = self.db.reconstruct(vector_id)
            query_hidden_states.append(query_hidden_state)
        query_hidden_states = np.stack(query_hidden_states)

        # MultilingualFaissDataset uses the following; not sure the reason.
        # faiss.ParameterSpace().set_index_parameter(self.db, "nprobe", 10)
        _, index = self.db.search(query_hidden_states, retri_factor)
        outputs = []
        for sample_idx, sample in enumerate(index):
            # the first video_id is always the video itself.
            cands = [video_ids[sample_idx]]
            for vector_idx in sample:
                if vector_idx >= 0 \
                        and vector_ids[sample_idx] != vector_idx:
                    cands.append(
                        self.vectoridx_to_videoid[vector_idx]
                    )
            outputs.append(cands)
        return outputs


class MMVectorRetriever(VectorRetrieverDM):
    """
    multimodal vector retriver:
    text retrieve video or video retrieve text.
    """

    def __init__(self, hidden_size, cent, db_type, examples_per_cent_to_train):
        super().__init__(
            hidden_size, cent, db_type, examples_per_cent_to_train)
        video_db = self.db
        super().__init__(
            hidden_size, cent, db_type, examples_per_cent_to_train)
        text_db = self.db
        self.db = {"video": video_db, "text": text_db}
        self.video_to_videoid = defaultdict(list)

    def __len__(self):
        assert self.db["video"].ntotal == self.db["text"].ntotal
        return self.db["video"].ntotal

    def make_direct_maps(self):
        faiss.downcast_index(self.db["video"]).make_direct_map()
        faiss.downcast_index(self.db["text"]).make_direct_map()

    def save(self, out_dir):
        faiss.write_index(
            self.db["video"],
            os.path.join(out_dir, "video_faiss_idx")
        )
        faiss.write_index(
            self.db["text"],
            os.path.join(out_dir, "text_faiss_idx")
        )

        with open(
                os.path.join(
                    out_dir, "videoid_to_vectoridx.pkl"),
                "wb") as fw:
            pickle.dump(
                self.videoid_to_vectoridx, fw,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def load(self, out_dir):
        fn = os.path.join(out_dir, "video_faiss_idx")
        video_db = faiss.read_index(fn)
        fn = os.path.join(out_dir, "text_faiss_idx")
        text_db = faiss.read_index(fn)
        self.db = {"video": video_db, "text": text_db}
        with open(
                os.path.join(out_dir, "videoid_to_vectoridx.pkl"), "rb") as fr:
            self.videoid_to_vectoridx = pickle.load(fr)
        self.video_to_videoid = defaultdict(list)

    def add(self, hidden_states, video_ids):
        """hidden_states is a pair `(video, text)`"""
        assert len(hidden_states) == len(video_ids), "{}, {}".format(
            str(len(hidden_states)), str(len(video_ids)))
        assert len(hidden_states.shape) == 3
        assert len(self.video_to_videoid) == 0

        valid_idx = []
        for idx, video_id in enumerate(video_ids):
            if video_id not in self.videoid_to_vectoridx:
                valid_idx.append(idx)
                self.videoid_to_vectoridx[video_id] = \
                    len(self.videoid_to_vectoridx)

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states[valid_idx]

        hidden_states = np.transpose(hidden_states, (1, 0, 2)).copy()
        if not self.db["video"].is_trained:
            self.train_cache.append(hidden_states)
            train_len = batch_size * len(self.train_cache)
            if train_len < self.train_thres:
                return

            hidden_states = np.concatenate(self.train_cache, axis=1)
            del self.train_cache
            self.db["video"].train(hidden_states[0, :self.train_thres])
            self.db["text"].train(hidden_states[1, :self.train_thres])
        self.db["video"].add(hidden_states[0])
        self.db["text"].add(hidden_states[1])

    def get_clips_by_video_id(self, video_id):
        if not self.video_to_videoid:
            for video_id, video_clip, text_clip in self.videoid_to_vectoridx:
                self.video_to_videoid[video_id].append(
                    (video_id, video_clip, text_clip))
        return self.video_to_videoid[video_id]

    def search(
        self,
        video_ids,
        target_modality,
        retri_factor=8
    ):
        if len(self.videoid_to_vectoridx) != len(self):
            raise ValueError(
                len(self.videoid_to_vectoridx),
                len(self)
            )

        if not self.make_direct_maps_done:
            self.make_direct_maps()
        if self.vectoridx_to_videoid is None:
            self.vectoridx_to_videoid = {
                self.videoid_to_vectoridx[videoid]: videoid
                for videoid in self.videoid_to_vectoridx
            }
            assert len(self.vectoridx_to_videoid) \
                == len(self.videoid_to_vectoridx)

        src_modality = "text" if target_modality == "video" else "video"

        query_hidden_states = []
        vector_ids = []
        for video_id in video_ids:
            vector_id = self.videoid_to_vectoridx[video_id]
            vector_ids.append(vector_id)
            query_hidden_state = self.db[src_modality].reconstruct(vector_id)
            query_hidden_states.append(query_hidden_state)
        query_hidden_states = np.stack(query_hidden_states)

        # MultilingualFaissDataset uses the following; not sure the reason.
        # faiss.ParameterSpace().set_index_parameter(self.db, "nprobe", 10)
        _, index = self.db[target_modality].search(
            query_hidden_states, retri_factor)
        outputs = []
        for sample_idx, sample in enumerate(index):
            cands = []
            for vector_idx in sample:
                if vector_idx >= 0:
                    cands.append(
                        self.vectoridx_to_videoid[vector_idx]
                    )
            outputs.append(cands)
        return outputs
