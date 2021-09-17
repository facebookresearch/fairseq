from cmath import log
from collections import Counter
import contextlib
from ctypes import util
from fairseq.tasks import online_backtranslation
import itertools
from fairseq.scoring.bleu import BleuConfig, Scorer
from fairseq.data import indexed_dataset
from multiprocessing import Pool, Value
from fairseq.binarizer import Binarizer
import functools
import time

from importlib_metadata import metadata
import numpy as np
from fairseq.dataclass.configs import FairseqConfig
import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.dataclass import utils as dataclass_utils
from fairseq.dataclass import configs
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from fairseq.logging import metrics, progress_bar
from omegaconf import DictConfig


from fairseq.options import get_parser, add_dataset_args, add_distributed_training_args
from fairseq.options import gen_parser_from_dataclass
from fairseq.options import CommonEvalConfig
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional
from fairseq.dataclass import FairseqDataclass
from sklearn.cluster import KMeans
from torch.nn import CosineSimilarity
import pickle

import torch.nn.functional as F
import torch.distributed as dist


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from .utils import object_to_tensor, tensor_to_object
from .utils import *


# NOTE -------------- GATHER SWAV PARALLEL DATA ------------------------------------------------
# These classes and functions responsible for aligning data from clusters to build pseudo-parallel data
# NOTE NOTICE:
#   - most source codes in this file are unnecessary for the final code, should be pruned
#   - the codes are kept here just for reference in the future


class SrcTgtMaybeWriter(object):
    def __init__(self, src_path, tgt_path, write) -> None:
        super().__init__()
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.write = write
        self.src = None
        self.tgt = None
        if self.write:
            root = "/".join(src_path.split('/')[:-1])
            os.makedirs(root, exist_ok=True)
            logger.warning(f'Writing data to tmp [src] {self.src_path}')
            logger.warning(f'Writing data to tmp [tgt] {self.tgt_path}')
            self.src = open(self.src_path, 'w', encoding='utf-8')
            self.tgt = open(self.tgt_path, 'w', encoding='utf-8')
    
    def __enter__(self):
        return self.src, self.tgt
    
    def __exit__(self, type, value, traceback):
        if self.src is not None:
            self.src.close()
        if self.tgt is not None:
            self.tgt.close()


# ---- Clustering Algorthms
class LangSepClusteringAlgorithm(object):
    def __init__(self, cfg, langs) -> None:
        super().__init__()
        self.cfg = cfg
        self.clu_cfg = cfg.swav_clustering
        self.langs = langs
    
    def compute_clusters(self, tensor_list, lang_list, **kwargs):
        raise NotImplementedError


class HardClusterLangSepClusAlgo(LangSepClusteringAlgorithm):
    def __init__(self, cfg, langs) -> None:
        super().__init__(cfg, langs)


class SoftClusterLangSepClusAlgo(LangSepClusteringAlgorithm):
    """Soft-Algo should return scores, not hard values"""
    def __init__(self, cfg, langs) -> None:
        super().__init__(cfg, langs)


class SoftmaxArgmaxLangSepClusAlgo(HardClusterLangSepClusAlgo):
    def compute_clusters(self, tensor_list, lang_list, **kwargs):
        assert all(x.size(-1) == tensor_list[0].size(-1) for x in tensor_list)
        n_clusters = tensor_list[0].size(1)
        clusters = []
        for i, (tensors, langs) in enumerate(zip(tensor_list, lang_list)):
            clu_data = tensors.softmax(-1)
            _lang_clusters = clu_data.argmax(-1)
            clusters.append(_lang_clusters)
        return clusters, n_clusters


class SoftmaxSoftLangSepClusAlgo(SoftClusterLangSepClusAlgo):
    def compute_clusters(self, tensor_list, lang_list, **kwargs):
        assert all(x.size(-1) == tensor_list[0].size(-1) for x in tensor_list)
        n_clusters = tensor_list[0].size(1)
        clusters = []
        for i, (tensors, langs) in enumerate(zip(tensor_list, lang_list)):
            clu_softmax = tensors.softmax(-1)
            clusters.append(clu_softmax)
        return clusters, n_clusters


# ---- Clustering aligners
class ClusteringAligner(object):
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__()
        self.cfg = cfg
        self.clu_cfg = cfg.swav_clustering
        self.langs = langs
        self.verify_clu_algo(clus_algo)
        self.clus_algo = clus_algo
        self.fwd_align_only = self.clu_cfg.fwd_align_only
        self.cross_align_n = self.clu_cfg.cross_align_n
        self.cross_align_threshold = self.clu_cfg.cross_align_threshold
        self.cross_count_stop = self.clu_cfg.cross_count_stop
        self.aligner_log_n = self.clu_cfg.aligner_log_n
        logger.warning(self)
    
    def verify_clu_algo(self, clus_algo):
        # virtually accept all cluster algorithms
        return
    
    @property
    def bilingual_only(self):
        return True
    
    @property
    def n_langs(self):
        return len(self.langs)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(fwd_only={self.fwd_align_only}, cross_align_n={self.cross_align_n}), '
            f'threshold={self.cross_align_threshold}, count_stop={self.cross_count_stop}'
        )
    
    def log_samples(self, data_pack, bpe=None, **kwargs):
        world_size, rank, rank_reprs = infer_dist_params(**kwargs)
        if bpe is None:
            bpe = lambda x: x
        keys = list(data_pack)
        for i in range(min(self.aligner_log_n, len(data_pack[keys[0]]))):
            logger.warning(f'{rank_reprs} [{i}] ALigned data: \n' + 
                "\n".join(f'\t[{k}]: {bpe(data_pack[k][i])}' for k in keys)
            )
    
    def align_data(self, bszs, ids_l, langs_l, texts_l, clusters_l, n_clusters, **kwargs):
        raise NotImplementedError
    
    def compute_clusters(self, tensor_list, lang_list, **kwargs):
        return self.clus_algo.compute_clusters(tensor_list, lang_list, **kwargs)
    

class BiParaHardClusteringAligner(ClusteringAligner):
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__(cfg, langs, clus_algo)
        assert len(langs) == 2, f'> 2 langs not supported, got {len(langs)}'
    
    def verify_clu_algo(self, clus_algo):
        assert isinstance(clus_algo, HardClusterLangSepClusAlgo), f'{clus_algo}'
    
    def _align(self, src, tgt, clusters_l, texts_l, src_f=None, tgt_f=None, **kwargs):
        raise NotImplementedError(f'Need to reimplement to update codes from the soft version')
        src_clusters = clusters_l[src[0]]
        tgt_clusters = clusters_l[tgt[0]]
        src_texts = texts_l[src[0]]
        tgt_texts = texts_l[tgt[0]]
        assert len(src_texts) == src_clusters.size(0), f'{len(src_texts)=} != {src_clusters.size()}'
        assert len(tgt_texts) == tgt_clusters.size(0), f'{len(tgt_texts)=} != {tgt_clusters.size()}'
        # src_unique_clus = torch.unique(src_clusters).tolist()
        src_data = []
        tgt_data = []
        for i, (src_clu_id, src_txt) in enumerate(zip(src_clusters, src_texts)):
            src_clu_id = src_clu_id.item()
            tgt_mask = tgt_clusters == src_clu_id
            tgt_intra_idx = torch.arange(tgt_clusters.size(0))[tgt_mask]
            if len(tgt_intra_idx) == 0:
                continue
            intra_tgt_txt = [tgt_texts[j] for j in tgt_intra_idx.tolist()]
            tgt_txts = list(np.random.choice(
                intra_tgt_txt, min(self.cross_align_n, len(intra_tgt_txt)), replace=False))
            new_src_txts = [src_txt] * len(tgt_txts)
            new_tgt_txts = tgt_txts
            if src_f is not None and tgt_f is not None:
                src_f.writelines([x + '\n' for x in new_src_txts])
                tgt_f.writelines([x + '\n' for x in new_tgt_txts])
            src_data.extend(new_src_txts)
            tgt_data.extend(new_tgt_txts)
            if self.cross_count_stop > 1 and len(src_data) > self.cross_count_stop:
                logger.warning(f'Aligning {src=} --> {tgt=} early stopped: {len(src_data)=}')
                break
        data_pack = {f'{src[1]}': src_data, f'{tgt[1]}': tgt_data}
        if "filter" in kwargs:
            data_pack = kwargs['filter'](data_pack, src[1], tgt[1], **kwargs)
        return data_pack
    
    def align_data(self, bszs, ids_l, langs_l, texts_l, clusters_l, n_clusters, **kwargs):
        """
        Most simple alignment method:
        * Align data from 2 languages langs[src, tgt]
        * clusters are hard, with no positional relationship information
            * TODO nxphi: Need to do another version where most adjacent inter-cluster samples
                can be obtain.
        If fwd_align_only: then only src->tgt and and reverse it for tgt-src
        else not: do pivot each language and align to the other language (data is more rich)
        Return dict {
            "{src}": ["hello", "thank you"],
            "{tgt}": ["bonjour", "merci"],
        }
        """
        raise NotImplementedError(f'nxphi: update this in accordance to soft version BiParaSoftScoreClusteringAligner.')
        assert len(bszs) == self.n_langs, f'only support 2 langs for now., {bszs=}'
        assert len(langs_l) == self.n_langs, f'only support 2 langs for now, {langs_l=}'
        assert isinstance(clusters_l[0], torch.Tensor), f'clus not torch.tensor: {type(clusters_l[0])=}'
        world_size, rank, rank_reprs = infer_dist_params(**kwargs)

        # align src to target
        src, tgt = (0, self.langs[0], langs_l[0][0]), (1, self.langs[1], langs_l[1][0]), 
        data_pack = {}
        with SrcTgtMaybeWriter(
            src_path=kwargs.get('src_path', None), tgt_path=kwargs.get('tgt_path', None),
            write=kwargs.get('write', False)
        ) as (src_f, tgt_f):
            logger.info(f'{rank_reprs} Aligning {src=} --> {tgt=}')
            data_pack = self._align(src, tgt, clusters_l, texts_l, src_f, tgt_f, **kwargs)
            logger.info(f'{rank_reprs} Aligned {src[1]=} --> {tgt[1]=} done:')
            self.log_samples(data_pack, **kwargs)
            if not self.fwd_align_only:
                logger.info(f'{rank_reprs} Aligning {tgt=} --> {src=}')
                bwd_data_pack = self._align(tgt, src, clusters_l, texts_l, tgt_f, src_f, **kwargs)
                logger.info(f'{rank_reprs} Aligned {tgt[1]=} --> {src[1]=} done: \n')
                self.log_samples(data_pack, **kwargs)
                for k, v in bwd_data_pack.items():
                    data_pack[k] = data_pack[k] + v
        return data_pack


class BiParaSoftScoreClusteringAligner(ClusteringAligner):
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__(cfg, langs, clus_algo)
        assert len(langs) == 2, f'> 2 langs not supported, got {len(langs)}'
        # TODO nxphi: add more constraint
        self.without_replacement = self.clu_cfg.without_replacement
    
    def maybe_to_cuda(self, val):
        # move to cuda to be faster
        return utils.move_to_cuda(val)
    
    def verify_clu_algo(self, clus_algo):
        assert isinstance(clus_algo, SoftClusterLangSepClusAlgo), f'{clus_algo}'
    
    def _acquire_align_indices(src_clu, tgt_clusters, selected_mask, **kwargs):
        """Need to return the indices of target that fit the alignment"""
        raise NotImplementedError
    
    def _align(self, src, tgt, clusters_l, texts_l, src_f=None, tgt_f=None, **kwargs):
        """Select tgt with priority base on softmax scores
        Distributed aligner: need to concat and obtain 
        """
        # TODO: need to do faster, currently already using CUDA
        # src_clusters = self.maybe_to_cuda(clusters_l[src[0]])
        # tgt_clusters = self.maybe_to_cuda(clusters_l[tgt[0]])
        raise NotImplementedError(f'Need to reimplement to update codes from the faster version')
        src_clusters = clusters_l[src[0]]
        tgt_clusters = clusters_l[tgt[0]]
        src_texts = texts_l[src[0]]
        tgt_texts = texts_l[tgt[0]]
        assert src_clusters.dim() == 2, f'{src_clusters.size()=}'
        assert len(src_texts) == src_clusters.size(0), f'{len(src_texts)=} != {src_clusters.size()}'
        assert len(tgt_texts) == tgt_clusters.size(0), f'{len(tgt_texts)=} != {tgt_clusters.size()}'
        world_size, rank, rank_reprs = infer_dist_params(**kwargs)
        is_dist = world_size > 1
        assert not is_dist, f'dist not impl {world_size=}'

        # src_unique_clus = torch.unique(src_clusters).tolist()
        src_data = []
        tgt_data = []
        selected_mask = None
        for i, (src_clu, src_txt) in enumerate(zip(src_clusters, src_texts)):
            tgt_intra_idx, selected_mask = self._acquire_align_indices(src_clu, tgt_clusters, selected_mask)
            if len(tgt_intra_idx) == 0:
                continue
            tgt_txts = [tgt_texts[j] for j in tgt_intra_idx.tolist()]
            new_src_txts = [src_txt] * len(tgt_txts)
            new_tgt_txts = tgt_txts
            if src_f is not None and tgt_f is not None:
                src_f.writelines([x + '\n' for x in new_src_txts])
                tgt_f.writelines([x + '\n' for x in new_tgt_txts])
            src_data.extend([src_txt] * len(tgt_txts))
            tgt_data.extend(tgt_txts)
            if self.cross_count_stop > 1 and len(src_data) > self.cross_count_stop:
                logger.warning(f'{rank_reprs} Aligning {src=} --> {tgt=} early stopped: {len(src_data)=}')
                break
        data_pack = {
            f'{src[1]}': src_data, 
            f'{tgt[1]}': tgt_data
        }
        if "filter" in kwargs:
            data_pack = kwargs['filter'](data_pack, src[1], tgt[1], **kwargs)
        return data_pack
    
    def align_data(self, bszs, ids_l, langs_l, texts_l, clusters_l, n_clusters, **kwargs):
        """
        FIXME: duplicate code above, because it is inheriting the base class without implementation
        Most simple alignment method:
        * Align data from 2 languages langs[src, tgt]
        * clusters are hard, with no positional relationship information
            * TODO nxphi: Need to do another version where most adjacent inter-cluster samples
                can be obtain.
        If fwd_align_only: then only src->tgt and and reverse it for tgt-src
        else not: do pivot each language and align to the other language (data is more rich)
        Return dict {
            "{src}": ["hello", "thank you"],
            "{tgt}": ["bonjour", "merci"],
        }
        """
        assert len(bszs) == self.n_langs, f'only support 2 langs for now., {bszs=}'
        assert len(langs_l) == self.n_langs, f'only support 2 langs for now, {langs_l=}'
        assert isinstance(clusters_l[0], torch.Tensor), f'clus not torch.tensor: {type(clusters_l[0])=}'
        world_size, rank, rank_reprs = infer_dist_params(**kwargs)

        # align src to target
        src, tgt = (0, self.langs[0], langs_l[0][0]), (1, self.langs[1], langs_l[1][0]), 

        data_pack = {}
        with SrcTgtMaybeWriter(
            src_path=kwargs.get('src_path', None), tgt_path=kwargs.get('tgt_path', None),
            write=kwargs.get('write', False)
        ) as (src_f, tgt_f):
            logger.info(f'{rank_reprs} Aligning {src=} --> {tgt=}')
            data_pack = self._align(src, tgt, clusters_l, texts_l, src_f, tgt_f, **kwargs)
            logger.info(f'{rank_reprs} Aligned {src[1]=} --> {tgt[1]=} done:')
            self.log_samples(data_pack, **kwargs)
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f'{rank_reprs}: Failure to empty cuda cache')
            if not self.fwd_align_only:
                logger.info(f'{rank_reprs} Reverse-Aligning {tgt=} --> {src=}')
                bwd_data_pack = self._align(tgt, src, clusters_l, texts_l, tgt_f, src_f, **kwargs)
                logger.info(f'{rank_reprs} Reverse-Aligned {tgt[1]=} --> {src[1]=} done: \n')
                self.log_samples(bwd_data_pack, **kwargs)
                for k, v in bwd_data_pack.items():
                    data_pack[k] = data_pack[k] + v
        return data_pack


class FasterBiParaSoftScoreClusteringAligner(BiParaSoftScoreClusteringAligner):
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__(cfg, langs, clus_algo)
        assert not self.without_replacement, f'with_replacement not suppoted'
        self.align_bsz = self.clu_cfg.align_bsz
    
    def _batch_acquire_align_indices(self, src_clu, tgt_clusters, selected_mask, **kwargs):
        raise NotImplementedError

    def _align(self, src, tgt, clusters_l, texts_l, src_f=None, tgt_f=None, **kwargs):
        """Select tgt with priority base on softmax scores
        do faster alignments, only support with_replacement
        Distributed aligning: gather targets from all workers for each src sentences
        # update version
        # Compute process-wise topk first and run distributed topk on top of the results
        # *** MAY HAVE TO USE point-to-point communication

        note: similarity scores: higher more similar
        """
        # TODO: need to do faster, currently already using CUDA
        src_clusters = clusters_l[src[0]].cpu()
        tgt_clusters = clusters_l[tgt[0]].cpu()
        src_texts = texts_l[src[0]]
        tgt_texts = texts_l[tgt[0]]
        assert src_clusters.dim() == 2, f'{src_clusters.size()=}'
        assert len(src_texts) == src_clusters.size(0), f'{len(src_texts)=} != {src_clusters.size()}'
        assert len(tgt_texts) == tgt_clusters.size(0), f'{len(tgt_texts)=} != {tgt_clusters.size()}'
        world_size, rank = kwargs.get('world_size', 1), kwargs.get('rank', 0)
        is_dist = world_size > 1
        # distributed 

        dist_tgt_clusters = dist_cat(world_size, tgt_clusters)
        dist_tgt_texts = dist_list(world_size, tgt_texts, 
            cat_fn=lambda xl: list(itertools.chain.from_iterable(xl)))
        assert len(dist_tgt_texts) == dist_tgt_clusters.size(0), f'{len(dist_tgt_texts)=} != {dist_tgt_clusters.size(0)=}'
        logger.warning(f'{self.__class__.__name__}[{rank}/{world_size}]: dist tgt: {tgt_clusters.size()} -> {dist_tgt_clusters.size()}')
            

        # old codes...........................
        src_data = []
        tgt_data = []
        selected_mask = None
        start = 0
        stop = False
        all_sim_scores = []
        while start < src_clusters.size(0) and not stop:
            src_clu_b = src_clusters[start: start + self.align_bsz]
            src_txt_b = src_texts[start: start + self.align_bsz]
            tgt_intra_idx_b, selected_mask, sim_scores = self._batch_acquire_align_indices(
                src_clu_b, dist_tgt_clusters, selected_mask)
            for i, (src_clu, src_txt, tgt_intra_idx, sim_sc) in enumerate(zip(
                src_clu_b, src_txt_b, tgt_intra_idx_b, sim_scores)):
                tgt_txts = [dist_tgt_texts[j] for j in tgt_intra_idx.tolist()]
                new_src_txts = [src_txt] * len(tgt_txts)
                new_tgt_txts = tgt_txts
                if src_f is not None and tgt_f is not None:
                    src_f.writelines(map(lambda x: x + '\n', new_src_txts))
                    tgt_f.writelines(map(lambda x: x + '\n', new_tgt_txts))
                src_data.extend([src_txt] * len(tgt_txts))
                tgt_data.extend(tgt_txts)
                all_sim_scores.append(sim_sc)
                if self.cross_count_stop > 1 and len(src_data) > self.cross_count_stop:
                    logger.warning(f'Aligning[{rank}/{world_size}] {src=} --> {tgt=} early stopped: {len(src_data)=}')
                    stop = True
                    break
            start += src_clu_b.size(0)
        all_sim_scores = torch.cat(all_sim_scores, 0)
        assert all_sim_scores.size(0) == len(src_data), f'{all_sim_scores.size(0)=} != {len(src_data)=}'
        data_pack = {
            f'{src[1]}': src_data, 
            f'{tgt[1]}': tgt_data
        }
        if "filter" in kwargs:
            data_pack = kwargs['filter'](data_pack, src[1], tgt[1], sim_scores=all_sim_scores, **kwargs)
        return data_pack


class DistBroadcastFilBiParaSoftScoreClusteringAligner(BiParaSoftScoreClusteringAligner):
    """DistBroadCast with onspot filtering
    This need to include the src side indices as well
    """
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__(cfg, langs, clus_algo)
        assert not self.without_replacement, f'with_replacement not suppoted'
        self.align_bsz = self.clu_cfg.align_bsz
        self._onspot_pre_filterer = build_onspot_pre_filterer(cfg, langs)
        self._onspot_post_filterer = build_onspot_post_filterer(cfg, langs)
        self._onspot_margin_filterer = build_onspot_margin_filterer(cfg, langs)
    
    def onspot_pre_filter(self, src_txts, tgt_txts, sims, **kwargs):
        return self._onspot_pre_filterer.onspot_filter(src_txts, tgt_txts, sims, **kwargs)
    
    def onspot_post_filter(self, srcs, tgts, sims, **kwargs):
        return self._onspot_post_filterer.onspot_filter(srcs, tgts, sims, **kwargs)
    
    def onspot_margin_setup(self, all_sim_scores, **kwags):
        # all_sim_scores: [bsz, topk], can be isinf
        return self._onspot_margin_filterer.onspot_margin_setup(all_sim_scores, **kwags)
    
    def onspot_margin_filter(self, margin_info, srcs, tgts, sims, **kwargs):
        return self._onspot_margin_filterer.onspot_filter(margin_info, srcs, tgts, sims, **kwargs)
    
    def _batch_acquire_align_indices(self, 
        src_clu: torch.Tensor, 
        src_texts: List[str],
        tgt_clusters: torch.Tensor, 
        tgt_texts: List[str],
        selected_mask=None, **kwargs) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        # NOTE src_texts and tgt_texts is to support onspot_pre_filterer
        # tgt_intra_idx: should be -1 for discarded sentence and place at the end
        # sim_scores should be -inf for respective discard_sentences
        raise NotImplementedError
    
    def _processwise_align(self, src, tgt, src_clusters, src_texts, tgt_clusters, tgt_texts, selected_mask=None):
        start = 0
        stop = False
        tgt_al_indices = []
        _tgt_al_txts = []
        tgt_al_txts = []
        tgt_sim_scores = []
        while start < src_clusters.size(0) and not stop:
            src_clu_b = src_clusters[start: start + self.align_bsz]
            # src_clu_idx = torch.arange(start, start + src_clu_b.size(0))
            src_txt_b = src_texts[start: start + self.align_bsz]

            tgt_intra_idx_b, selected_mask, sim_scores = self._batch_acquire_align_indices(
                src_clu_b, src_txt_b, tgt_clusters, tgt_texts, selected_mask)
            # NOTE: tgt_intra_idx_b can be <0 for discarded sentences, then sim_sores = -inf
            tgt_al_indices.append(tgt_intra_idx_b)
            tgt_txts = [[tgt_texts[j] if j >= 0 else "<empty>" for j in tgt_intra_idx.tolist()]
                for i, tgt_intra_idx in enumerate(tgt_intra_idx_b)
            ]
            # _tgt_al_txts.append(tgt_txts)
            tgt_al_txts.extend(tgt_txts)
            tgt_sim_scores.append(sim_scores)
            start += src_clu_b.size(0)
        tgt_al_indices = torch.cat(tgt_al_indices, 0)
        tgt_sim_scores = torch.cat(tgt_sim_scores, 0)
        # tgt_al_txts = list(itertools.chain.from_iterable(_tgt_al_txts))
        assert len(tgt_al_txts) == src_clusters.size(0), f'{len(tgt_al_txts)=} != {src_clusters.size(0)=}'
        assert tgt_al_indices.size(0) == src_clusters.size(0), f'{tgt_al_indices.size(0)=} != {src_clusters.size(0)=}'
        assert tgt_sim_scores.size(0) == src_clusters.size(0), f'{tgt_sim_scores.size(0)=} != {src_clusters.size(0)=}'
        return tgt_al_indices, tgt_sim_scores, tgt_al_txts

    def _align(self, src, tgt, clusters_l, texts_l, src_f=None, tgt_f=None, **kwargs):
        """Select tgt with priority base on softmax scores
        Distributed aligning: 
        steps:
        0. Prepare max-size of all tgt_clusters and tgt_texts
        1. for each r in world_size:
            1.1 broadcast and recv data from r -> to other processes
            1.2 Process data and topk reduce for each r
        2. global reduce on the reduced topk

        note: similarity scores: higher more similar
        """
        # TODO: need to do faster, currently already using CUDA
        src_clusters = clusters_l[src[0]].cpu()
        tgt_clusters = clusters_l[tgt[0]].cpu()
        src_texts = texts_l[src[0]]
        tgt_texts = texts_l[tgt[0]]
        assert src_clusters.dim() == 2, f'{src_clusters.size()=}'
        assert len(src_texts) == src_clusters.size(0), f'{len(src_texts)=} != {src_clusters.size()}'
        assert len(tgt_texts) == tgt_clusters.size(0), f'{len(tgt_texts)=} != {tgt_clusters.size()}'
        world_size, rank = kwargs.get('world_size', 1), kwargs.get('rank', 0)
        rank_reprs = f'[{rank}/{world_size}]'
        srctgt = f'[{src[1]}{src[0]}->{tgt[1]}{tgt[0]}]'
        is_dist = world_size > 1

        all_src_data = []
        all_tgt_data = []
        all_sim_scores = []
        if is_dist:
            # distributed 
            # step 0: prepare dist sizes
            cur_device = tgt_clusters.device
            d_clu_bszs = dist_batch_sizes(world_size, tgt_clusters, dim=0)
            txt_tensor, txt_local_size = object_to_tensor(tgt_texts)
            d_txt_bsizes = utils.move_to_cpu(dist_list(world_size, utils.move_to_cuda(txt_local_size), same_size=True))
            
            # step 1 (version 2) - cleaner, but need to check
            logger.warning(f'{rank_reprs}{srctgt} Start iterative_broadcast_process')
            torch.cuda.empty_cache()
            _tgt_al_indices_l, dist_sim_scores, _dist_tgt_al_txts = iterative_broadcast_process(
                rank=rank, world_size=world_size,
                tensors=[tgt_clusters, txt_tensor],
                sizes=[[torch.tensor([x.item(), tgt_clusters.size(1)]).long() for x in d_clu_bszs], 
                        d_txt_bsizes],
                proc_fn=lambda _r, _ins: self._processwise_align(
                        src, tgt, src_clusters, src_texts, 
                        tgt_clusters=_ins[0], 
                        tgt_texts=tensor_to_object(utils.move_to_cpu(_ins[1]).type(torch.ByteTensor), d_txt_bsizes[_r]),
                    ),
            )

            # step 2: aggregate tgt data from all processes and topk again
            dist_sim_scores = torch.cat(dist_sim_scores, 1)
            dist_tgt_al_txts = [
                list(itertools.chain.from_iterable(_dist_tgt_al_txts[j][i] for j in range(world_size)))
                for i in range(dist_sim_scores.size(0))
            ]
            logger.warning(f'{rank_reprs}{srctgt} Finished broadcast and processing of all processes, start aggregating: {dist_sim_scores.size()=}')
            del _dist_tgt_al_txts
            assert all(len(x) == dist_sim_scores.size(1) for x in dist_tgt_al_txts)
            dist_avail_indices = torch.arange(dist_sim_scores.size(1), device=dist_sim_scores.device)

            min_topk = min(self.cross_align_n, dist_avail_indices.size(0))
            dist_topk_comp = torch.topk(dist_sim_scores, min_topk, dim=-1, largest=True)
            dist_topk_comp_indices = dist_topk_comp.indices.cpu()
            dist_topk_sim_values = dist_topk_comp.values.cpu()
            dist_selected_indices = dist_avail_indices.index_select(0, dist_topk_comp_indices.view(-1)).view(
                dist_topk_comp_indices.size()
            )
        else:
            raise NotImplementedError(f'world size {world_size} not supported, need distributed')
        margin_info = self.onspot_margin_setup(dist_topk_sim_values)
        
        logger.warning(f'{rank_reprs}{srctgt} start add data: {dist_topk_sim_values.size()=}, margin: {margin_info}')

        for j in range(dist_selected_indices.size(1)):
            # each topk
            tgt_topk_txts = [
                tgt_txts[dist_selected_indices[i, j].item()]
                for i, tgt_txts in enumerate(dist_tgt_al_txts)
            ]
            topk_sim_scores = dist_topk_sim_values[:, j]
            assert len(tgt_topk_txts) == len(src_texts)
            # onspot_post_filterer
            f_src_txts, f_tgt_txts, f_sim_scores = self.onspot_post_filter(src_texts, tgt_topk_txts, topk_sim_scores)
            f_src_txts, f_tgt_txts, f_sim_scores = self.onspot_margin_filter(margin_info, f_src_txts, f_tgt_txts, f_sim_scores)

            logger.warning(f'{rank_reprs}{srctgt} onspose_post_filter: retrain {len(f_src_txts)} / {len(src_texts)} ({len(f_src_txts)/float(len(src_texts))})')
            if len(f_src_txts) <= 0:
                continue
            assert len(f_src_txts) == len(f_tgt_txts)
            assert len(f_src_txts) == f_sim_scores.size(0)
            all_src_data.extend(f_src_txts)
            all_tgt_data.extend(f_tgt_txts)
            all_sim_scores.append(f_sim_scores)
            if src_f is not None and tgt_f is not None:                
                src_f.writelines(map(lambda x: x + '\n', f_src_txts))
                tgt_f.writelines(map(lambda x: x + '\n', f_tgt_txts))
            if self.cross_count_stop > 1 and len(all_src_data) > self.cross_count_stop:
                logger.warning(f'Aligning[{rank}/{world_size}] {src=} --> {tgt=} early stopped: {len(all_src_data)=}')
                break
        assert len(all_sim_scores) > 0, f'sim_scores empty, looks like post filterer has filtered all pairs'
        all_sim_scores = torch.cat(all_sim_scores, 0)
        assert all_sim_scores.size(0) == len(all_src_data), f'{all_sim_scores.size(0)=} != {len(all_src_data)=}'
        logger.warning(f'{rank_reprs}{srctgt}: finish aligner, stats: {all_sim_scores.mean()=}, {all_sim_scores.max()=}, {all_sim_scores.min()=}')
        data_pack = {
            f'{src[1]}': all_src_data, 
            f'{tgt[1]}': all_tgt_data
        }
        if "filter" in kwargs:
            data_pack = kwargs['filter'](data_pack, src[1], tgt[1], sim_scores=all_sim_scores, **kwargs)
        return data_pack



class DiffNormBiParaClusteringAligner(BiParaSoftScoreClusteringAligner):
    """
    Rank selection by l2 norm of difference of src and tgt
    """
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__(cfg, langs, clus_algo)
        self.diff_norm_mode = self.clu_cfg.diff_norm_mode

    def _acquire_align_indices(self, src_clu, tgt_clusters, selected_mask, **kwargs):
        avail_clusters = tgt_clusters
        avail_indices = torch.arange(tgt_clusters.size(0), device=avail_clusters.device)
        if self.without_replacement and selected_mask is not None:
            avail_clusters = avail_clusters[~selected_mask]
            avail_indices = avail_indices[~selected_mask]
        if avail_indices.size(0) == 0:
            # avail empty, return empty indices
            return avail_indices, selected_mask
        if src_clu.dim() == 1:
            src_clu = src_clu.unsqueeze(0)
        min_topk = min(self.cross_align_n, avail_clusters.size(0))
        com_values = torch.linalg.norm(src_clu - avail_clusters, self.diff_norm_mode, dim=-1)
        topk_comp = torch.topk(-com_values, min_topk, dim=-1, largest=True)
        topk_comp_indices = topk_comp.indices
        topk_sim_values = topk_comp.values
        selected_indices = avail_indices.index_select(0, topk_comp_indices)
        
        if selected_mask is not None:
            selected_mask = selected_mask.scatter_(0, selected_indices, 1)
        return selected_indices, selected_mask, topk_sim_values


class DiffNormFasterBiParaClusteringAligner(FasterBiParaSoftScoreClusteringAligner):
    """Eucledian distance"""
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__(cfg, langs, clus_algo)
        self.align_bsz = self.clu_cfg.align_bsz
        self.diff_norm_mode = self.clu_cfg.diff_norm_mode
        assert not self.without_replacement, f'with_replacement not suppoted'

    @torch.no_grad()
    def _batch_acquire_align_indices(self, src_clu, tgt_clusters, selected_mask, **kwargs):
        avail_clusters = tgt_clusters
        avail_size = avail_clusters.size(0)
        min_topk = min(self.cross_align_n, avail_clusters.size(0))
        avail_indices = torch.arange(tgt_clusters.size(0), device=avail_clusters.device)
        _src_clu = self.maybe_to_cuda(src_clu.unsqueeze(0))  # [sbsz, 1, dim]
        # avail_clusters = avail_clusters.unsqueeze(0)  # [1, tbsz, dim]
        # batch for target
        start = 0
        com_values = []
        while start < avail_size:
            # _avail_clus_batch = self.maybe_to_cuda(avail_clusters[start: start + self.align_bsz]).unsqueeze_(0)
            _avail_clus_batch = self.maybe_to_cuda(avail_clusters[start: start + self.align_bsz]).unsqueeze_(0)
            # _com_values = torch.linalg.norm(_src_clu - _avail_clus_batch, self.diff_norm_mode, dim=-1)
            _com_values = torch.cdist(_src_clu, _avail_clus_batch, p=self.diff_norm_mode).squeeze_(0)
            com_values.append(_com_values)
            start += _avail_clus_batch.size(1)
        com_values = torch.cat(com_values, 1)
        # com_values [sbzs, tbsz]
        topk_comp = torch.topk(com_values, min_topk, dim=-1, largest=False)
        topk_comp_indices = topk_comp.indices.cpu()     # [sbzs, topk]
        topk_sim_values = -topk_comp.values.cpu()       # [sbzs, topk]
        selected_indices = avail_indices.index_select(0, topk_comp_indices.view(-1)).view(
            topk_comp_indices.size()
        )
        return selected_indices, selected_mask, topk_sim_values


class DiffNormFilDistBrBiParaClusteringAligner(DistBroadcastFilBiParaSoftScoreClusteringAligner):
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__(cfg, langs, clus_algo)
        self.align_bsz = self.clu_cfg.align_bsz
        self.diff_norm_mode = self.clu_cfg.diff_norm_mode
        assert not self.without_replacement, f'with_replacement not suppoted'
    
    @torch.no_grad()
    def _batch_acquire_align_indices(self,
        src_clu: torch.Tensor, 
        src_texts: List[str],
        tgt_clusters: torch.Tensor, 
        tgt_texts: List[str],
        selected_mask=None, **kwargs) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        # NOTE src_texts and tgt_texts is to support onspot_pre_filterer
        # tgt_intra_idx: should be -1 for discarded sentence and place at the end
        # sim_scores should be -inf for respective discard_sentences
        avail_clusters = tgt_clusters
        avail_size = avail_clusters.size(0)
        min_topk = min(self.cross_align_n, avail_clusters.size(0))
        avail_indices = torch.arange(tgt_clusters.size(0), device=avail_clusters.device)
        # _src_clu = self.maybe_to_cuda(src_clu.unsqueeze(1))  # [sbsz, 1, dim]
        _src_clu = self.maybe_to_cuda(src_clu.unsqueeze(0))  # [1, sbsz, dim]
        # batch for target
        start = 0
        com_values = []
        while start < avail_size:
            _avail_clus_b = self.maybe_to_cuda(avail_clusters[start: start + self.align_bsz]).unsqueeze_(0)
            _avail_txts_b = tgt_texts[start: start + self.align_bsz]
            # _com_values = torch.linalg.norm(_src_clu - _avail_clus_batch, self.diff_norm_mode, dim=-1)
            _com_values = -torch.cdist(_src_clu, _avail_clus_b, p=self.diff_norm_mode).squeeze_(0)
            _srcs, _tgts, _com_values = self.onspot_pre_filter(src_texts, _avail_txts_b, _com_values)
            com_values.append(_com_values)
            start += _avail_clus_b.size(1)
        com_values = torch.cat(com_values, 1)
        # com_values [sbzs, tbsz]
        topk_comp = torch.topk(com_values, min_topk, dim=-1, largest=True)
        topk_comp_indices = topk_comp.indices.cpu()     # [sbzs, topk]
        topk_sim_values = topk_comp.values.cpu()       # [sbzs, topk]
        selected_indices = avail_indices.index_select(0, topk_comp_indices.view(-1)).view(
            topk_comp_indices.size()
        )
        return selected_indices, selected_mask, topk_sim_values


class CosineSimFasterBiParaClusteringAligner(FasterBiParaSoftScoreClusteringAligner):
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__(cfg, langs, clus_algo)
        self.align_bsz = self.clu_cfg.align_bsz
        assert not self.without_replacement, f'with_replacement not suppoted'

    @torch.no_grad()
    def _batch_acquire_align_indices(self, src_clu, tgt_clusters, selected_mask, **kwargs):
        avail_clusters = tgt_clusters
        avail_size = avail_clusters.size(0)
        min_topk = min(self.cross_align_n, avail_clusters.size(0))
        avail_indices = torch.arange(tgt_clusters.size(0), device=avail_clusters.device)
        _src_clu = self.maybe_to_cuda(src_clu.unsqueeze(1))  # [sbsz, 1, dim]
        # avail_clusters = avail_clusters.unsqueeze(0)  # [1, tbsz, dim]
        # batch for target
        start = 0
        com_values = []
        while start < avail_size:
            _avail_clus_batch = self.maybe_to_cuda(avail_clusters[start: start + self.align_bsz]).unsqueeze_(0)
            _com_values = F.cosine_similarity(_src_clu, _avail_clus_batch, dim=-1)
            com_values.append(_com_values)
            start += _avail_clus_batch.size(1)
        com_values = torch.cat(com_values, 1)

        # compute cosine similarity
        # com_values [sbzs, tbsz]
        topk_comp = torch.topk(com_values, min_topk, dim=-1, largest=True)
        topk_comp_indices = topk_comp.indices.cpu()   # [sbzs, topk]
        topk_sim_values = topk_comp.values.cpu()       # [sbzs, topk]
        
        selected_indices = avail_indices.index_select(0, topk_comp_indices.view(-1)).view(
            topk_comp_indices.size())
        return selected_indices, selected_mask, topk_sim_values


class CosineSimFilDistBrBiParaClusteringAligner(DistBroadcastFilBiParaSoftScoreClusteringAligner):
    def __init__(self, cfg, langs, clus_algo: LangSepClusteringAlgorithm) -> None:
        super().__init__(cfg, langs, clus_algo)
        self.align_bsz = self.clu_cfg.align_bsz
        assert not self.without_replacement, f'with_replacement not suppoted'

    @torch.no_grad()
    def _batch_acquire_align_indices(self, src_clu, tgt_clusters, selected_mask, **kwargs):
        avail_clusters = tgt_clusters
        avail_size = avail_clusters.size(0)
        min_topk = min(self.cross_align_n, avail_clusters.size(0))
        avail_indices = torch.arange(tgt_clusters.size(0), device=avail_clusters.device)
        _src_clu = self.maybe_to_cuda(src_clu.unsqueeze(1))  # [sbsz, 1, dim]
        # avail_clusters = avail_clusters.unsqueeze(0)  # [1, tbsz, dim]
        # batch for target
        start = 0
        com_values = []
        while start < avail_size:
            _avail_clus_batch = self.maybe_to_cuda(avail_clusters[start: start + self.align_bsz]).unsqueeze_(0)
            _com_values = F.cosine_similarity(_src_clu, _avail_clus_batch, dim=-1)
            com_values.append(_com_values)
            start += _avail_clus_batch.size(1)
        com_values = torch.cat(com_values, 1)

        # compute cosine similarity
        # com_values [sbzs, tbsz]
        topk_comp = torch.topk(com_values, min_topk, dim=-1, largest=True)
        topk_comp_indices = topk_comp.indices.cpu()   # [sbzs, topk]
        topk_sim_values = topk_comp.values.cpu()       # [sbzs, topk]
        
        selected_indices = avail_indices.index_select(0, topk_comp_indices.view(-1)).view(
            topk_comp_indices.size())
        return selected_indices, selected_mask, topk_sim_values


class SoftmaxArgmaxRankFasterBiParaCluAligner(FasterBiParaSoftScoreClusteringAligner):
    @torch.no_grad()
    def _batch_acquire_align_indices(self, src_clu, tgt_clusters, selected_mask, **kwargs):
        raise NotImplementedError(f'Need to follow procedure as CosineSimFasterBiParaClusteringAligner')
        avail_clusters = tgt_clusters
        avail_indices = torch.arange(tgt_clusters.size(0), device=avail_clusters.device)
        src_clu = src_clu.unsqueeze(1)  # [sbsz, 1, dim]
        src_clu = src_clu.softmax(-1).argmax(-1)
        avail_clusters = tgt_clusters.unsqueeze(0)  # [1, tbsz, dim]
        avail_clus_argmax = avail_clusters.softmax(-1).argmax(-1, keepdim=True)
        avail_clus_arg_val = avail_clusters.index_select(-1, avail_clus_argmax).squeeze_(-1)
        avail_clus_argmax = avail_clus_argmax.squeeze_(-1)
        src_tgt_mask = src_clu == avail_clus_argmax # [sbsz, tbsz]
        avail_clus_arg_val = avail_clus_arg_val.masked_fill_(~src_tgt_mask, float('inf'))
        min_topk = min(self.cross_align_n, avail_clusters.size(0))
        topk_comp = torch.topk(avail_clus_arg_val, min_topk, dim=-1, largest=False)
        topk_comp_indices = topk_comp.indices   # [sbzs, topk]
        topk_comp_values = topk_comp.values.cpu()       # [sbzs, topk]

        selected_indices = avail_indices.index_select(0, topk_comp_indices.view(-1)).view(
            topk_comp_indices.size())
        return selected_indices, selected_mask, topk_comp_values


# ------- Filter for aligned data


class OnSpotFilterer(object):
    def __init__(self, cfg, langs) -> None:
        super().__init__()
        self.cfg = cfg
        self.clu_cfg = cfg.swav_clustering
        self.langs = langs
        self.onspot_score_ninf = self.clu_cfg.onspot_score_ninf
        logger.info(f'Onspot filterer: {self.__class__.__name__}')
    
    def onspot_filter(self, *args, **kwargs):
        raise NotImplementedError


class OnSpotPreFilterer(OnSpotFilterer):
    def onspot_filter(self, srcs: List[str], tgts: List[str], sim_matrix: torch.Tensor, **kwargs):
        assert len(srcs) == sim_matrix.size(0), f'{len(srcs)=}!=0-{sim_matrix.size(0)}'
        assert len(tgts) == sim_matrix.size(1), f'{len(tgts)=}!=1-{sim_matrix.size(1)}'
        # assert -1 to sim_matrix that we want to filter
        return srcs, tgts, sim_matrix


class LenRatioOSPreFilterer(OnSpotPreFilterer):
    def __init__(self, cfg, langs) -> None:
        super().__init__(cfg, langs)
        self.onspot_len_ratio = self.clu_cfg.onspot_len_ratio
        assert (self.onspot_len_ratio is not None and self.onspot_len_ratio >= 1.0
            ), f'len_ratio: {self.onspot_len_ratio} invalid.'
    
    def onspot_filter(self, srcs: List[str], tgts: List[str], sim_matrix: torch.Tensor, **kwargs):
        assert len(srcs) == sim_matrix.size(0), f'{len(srcs)=}!=0-{sim_matrix.size(0)}'
        assert len(tgts) == sim_matrix.size(1), f'{len(tgts)=}!=1-{sim_matrix.size(1)}'
        # assert -inf to sim_matrix that we want to filter
        src_len = torch.tensor([len(x) for x in srcs]).float().to(sim_matrix.device)
        tgt_len = torch.tensor([len(x) for x in tgts]).float().to(sim_matrix.device)
        len_mat = src_len.unsqueeze(1) / tgt_len.unsqueeze(0)
        discard = (len_mat > self.onspot_len_ratio) | (len_mat < 1.0 / self.onspot_len_ratio)
        sim_matrix = sim_matrix.masked_fill_(discard, sim_matrix.new(1).fill_(self.onspot_score_ninf).squeeze_(0))
        return srcs, tgts, sim_matrix


class ThresLenRatioOSPreFilterer(OnSpotPreFilterer):
    def __init__(self, cfg, langs) -> None:
        super().__init__(cfg, langs)
        self.onspot_len_ratio = self.clu_cfg.onspot_len_ratio
        self.onspot_threshold = self.clu_cfg.onspot_threshold
        assert (self.onspot_len_ratio is not None and self.onspot_len_ratio >= 1.0
            ), f'len_ratio: {self.onspot_len_ratio} invalid.'
        assert isinstance(self.onspot_threshold, float), f'{self.onspot_threshold}'
    
    def onspot_filter(self, srcs: List[str], tgts: List[str], sim_matrix: torch.Tensor, **kwargs):
        assert len(srcs) == sim_matrix.size(0), f'{len(srcs)=}!=0-{sim_matrix.size(0)}'
        assert len(tgts) == sim_matrix.size(1), f'{len(tgts)=}!=1-{sim_matrix.size(1)}'
        # assert -inf to sim_matrix that we want to filter
        src_len = torch.tensor([len(x) for x in srcs]).float().to(sim_matrix.device)
        tgt_len = torch.tensor([len(x) for x in tgts]).float().to(sim_matrix.device)
        len_mat = src_len.unsqueeze(1) / tgt_len.unsqueeze(0)
        discard = (len_mat > self.onspot_len_ratio) | (len_mat < 1.0 / self.onspot_len_ratio)
        discard |= sim_matrix > self.onspot_threshold
        sim_matrix = sim_matrix.masked_fill_(discard, sim_matrix.new(1).fill_(self.onspot_score_ninf).squeeze_(0))
        return srcs, tgts, sim_matrix


class OnSpotPostFilterer(OnSpotFilterer):
    def onspot_filter(self, srcs: List[str], tgts: List[str], sims: torch.Tensor, **kwargs):
        # isclose = torch.isinf(sims, sims.new(1).fill_(self.onspot_score_ninf).squeeze_(0))
        isclose = torch.isinf(sims)
        if torch.any(isclose):
            indices = torch.arange(sims.size(0))[~isclose]
            _sims = sims[indices]
            _indices = indices.tolist()
            _srcs = list(map(lambda i: srcs[i], _indices))
            _tgts = list(map(lambda i: tgts[i], _indices))
            return _srcs, _tgts, _sims
        return srcs, tgts, sims


class OnSpotMarginFilterer(OnSpotFilterer):
    def onspot_margin_setup(self, sims: torch.Tensor, **kwargs):
        isinf = torch.isinf(sims)
        scores = sims[~isinf]
        margin_info = {
            "mean": scores.mean(),
            "min": scores.min(),
            "max": scores.max(),
            "ori_size:": sims.view(-1).size(0),
            "size": scores.size(0),
        }
        return margin_info

    def onspot_filter(self, margin_info: dict, srcs: List[str], tgts: List[str], sims: torch.Tensor, **kwargs):
        return srcs, tgts, sims


class GreaterAvgOnSpotMarginFilterer(OnSpotMarginFilterer):
    def onspot_filter(self, margin_info: dict, srcs: List[str], tgts: List[str], sims: torch.Tensor, **kwargs):
        mean = margin_info['mean']
        accept = sims >= mean
        if torch.any(accept):
            indices = torch.arange(sims.size(0))[accept]
            _sims = sims[indices]
            _indices = indices.tolist()
            _srcs = list(map(lambda i: srcs[i], _indices))
            _tgts = list(map(lambda i: tgts[i], _indices))
            return _srcs, _tgts, _sims
        return srcs, tgts, sims


class GreaterAvgRatioOnSpotMarginFilterer(OnSpotMarginFilterer):
    def __init__(self, cfg, langs) -> None:
        super().__init__(cfg, langs)
        self.onspot_margin_ratio = self.clu_cfg.onspot_margin_ratio

    def onspot_filter(self, margin_info: dict, srcs: List[str], tgts: List[str], sims: torch.Tensor, **kwargs):
        mean = margin_info['mean']
        # beware becase score can be negative, this means sims/mean would not reflect greater values
        assert sims.max() * sims.min() > 0, f'{sims.max()=} {sims.min()} different sign, invalid for this filterer'
        if sims.max() < 0:
            # logger.warning(f'sims negative')
            accept = (sims / mean) < (1.0 / self.onspot_margin_ratio)
        else:
            # logger.warning(f'sims positive')
            accept = (sims / mean) > self.onspot_margin_ratio
        if torch.any(accept):
            indices = torch.arange(sims.size(0))[accept]
            _sims = sims[indices]
            _indices = indices.tolist()
            _srcs = list(map(lambda i: srcs[i], _indices))
            _tgts = list(map(lambda i: tgts[i], _indices))
            return _srcs, _tgts, _sims
        return srcs, tgts, sims



def build_onspot_pre_filterer(cfg, langs=None):
    mode = cfg.swav_clustering.onspot_pre_filterer
    if mode is None or mode == 'default':
        return OnSpotPreFilterer(cfg, langs)
    elif mode == 'len_ratio':
        return LenRatioOSPreFilterer(cfg, langs)
    elif mode == 'thres_len_ratio':
        return LenRatioOSPreFilterer(cfg, langs)
    else:
        raise ValueError(f'mode {mode} not found')


def build_onspot_post_filterer(cfg, langs=None):
    mode = cfg.swav_clustering.onspot_post_filterer
    if mode is None or mode == 'default':
        return OnSpotPostFilterer(cfg, langs)
    else:
        raise ValueError(f'mode {mode} not found')


def build_onspot_margin_filterer(cfg, langs=None):
    mode = cfg.swav_clustering.onspot_margin_filterer
    if mode is None or mode == 'default':
        return OnSpotMarginFilterer(cfg, langs)
    elif mode == "greater_avg":
        return GreaterAvgOnSpotMarginFilterer(cfg, langs)
    elif mode == "greater_avg_ratio":
        return GreaterAvgRatioOnSpotMarginFilterer(cfg, langs)
    else:
        raise ValueError(f'mode {mode} not found')


class AlignedFilterer(object):
    def __init__(self, cfg, langs) -> None:
        super().__init__()
        self.cfg = cfg
        self.clu_cfg = cfg.swav_clustering
        self.langs = langs
        self.filter_mode = self.clu_cfg.filter_mode
        self.filter_model_path = self.clu_cfg.filter_model_path
        self.use_fp16 = cfg.common.fp16
        self.use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    
    def filter(self, data_pack, src, tgt, **kwargs):
        assert isinstance(data_pack, dict)
        assert all(l in data_pack for l in self.langs), f'{self.langs} != {data_pack.keys()}'
        logger.warning(f'{self.__class__.__name__} filtering...')
        return data_pack


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def decode_fn(x, bpe, tokenizer):
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


class UMTBleuAlignedFilterer(AlignedFilterer):
    def __init__(self, cfg, langs) -> None:
        super().__init__(cfg, langs)
        self.filter_model_path = self.clu_cfg.filter_model_path
        self.filter_model_task = self.clu_cfg.filter_model_task
        self.filter_bleu_max_tokens = self.clu_cfg.filter_bleu_max_tokens
        self.filter_bleu_threshold = self.clu_cfg.filter_bleu_threshold
        self.filter_bleu_order = self.clu_cfg.filter_bleu_order
        self._model = None
        self._task = None
        self._bleu_scorer = None
    
    @contextlib.contextmanager
    def load_eval_model_task(self, overrides: Optional[Dict[str, Any]] = None, force_reload=False):
        logger.info(f'Load eval model {self.filter_model_path}')
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [self.filter_model_path],
            arg_overrides=overrides,
            # suffix=cfg.checkpoint.checkpoint_suffix,
            # task=task,
        )
        model = models[0]

        # Move models to GPU
        for model in models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda:
                model.cuda()
        yield model, task
        del model, task
    
    def filter_fullfil(self, score, **kwargs):
        return score >= self.filter_bleu_threshold
    
    def filter_scores(self, srcs, tgts, scores):
        # NOTE this can be hard values or percentile
        assert len(srcs) == len(tgts), f'{len(srcs)=} != {len(tgts)=}'
        assert len(srcs) == len(scores), f'{len(srcs)=} != {len(scores)=}'
        f_srcs = []
        f_tgts = []
        ex_srcs = []
        ex_tgts = []
        for i, (_src, _tgt, _score) in enumerate(zip(srcs, tgts, scores)):
            _f_srcs, _f_tgts, _ex_srcs, _ex_tgts = self.filter_sent_score(_src, _tgt, _score)
            f_srcs.extend(_f_srcs)
            f_tgts.extend(_f_tgts)
            ex_srcs.extend(_ex_srcs)
            ex_tgts.extend(_ex_tgts)
        return f_srcs, f_tgts, ex_srcs, ex_tgts
    
    def filter_sent_score(self, src, tgt, score):
        if self.filter_fullfil(score):
            return [src], [tgt], [], []
        else:
            return [], [], [src], [tgt]
    
    def compute_scores(self, dictionary, hypos, tgts):
        scores = []
        # NOTE Switch the role of hyp and tgt:
        #   hypo is from the eval model, consider as ref
        #   tgt is from aligned data, consider as hyp
        for i, (ref, hyp) in enumerate(zip(hypos, tgts)):
            scores.append(self.compute_sent_score(dictionary, ref, hyp))
        return scores

    def compute_sent_score(self, dictionary, hyp, tgt):
        if self._bleu_scorer is None:
            bleu_config = BleuConfig(pad=dictionary.pad(), eos=dictionary.eos(), unk=dictionary.unk())
            bleu_scorer = Scorer(bleu_config)
            self._bleu_scorer = bleu_scorer
        # NOTE Switch the role of hyp and tgt:
        #   hypo is from the eval model, consider as ref
        #   tgt is from aligned data, consider as hyp
        self._bleu_scorer.reset()
        self._bleu_scorer.add(hyp, tgt)
        _score = self._bleu_scorer.score(min(len(hyp), len(tgt), self.filter_bleu_order))
        self._bleu_scorer.reset()
        return _score
    
    def _generate_score_n_filter(self, model, task, pair_dataset, src, tgt, batch_size=None, cut_off=1e15, 
        scorer=None, filterer=None, src_file=None, tgt_file=None, 
        **kwargs
    ):
        world_size, rank, rank_reprs = infer_dist_params(**kwargs)
        # NOTE: if run with clustering: data is already sharded
        #       if run barely filter:   data is NOT sharded so we have to pass the num_shards and shard_id into it
        num_shards = kwargs.get('num_shards', 1)
        shard_id = kwargs.get('shard_id', 0)
        write_to_tmp = (src_file is not None) and (tgt_file is not None)
        logger.warning(f'{rank_reprs}: generate data and (maybe) filter: shard {shard_id}/{num_shards}, write tmp: {write_to_tmp}')
        cfg = self.cfg
        dictionary = task.source_dictionary
        max_positions = utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in [model]]
        )
        iterator = task.get_batch_iterator(
            dataset=pair_dataset,
            max_tokens=self.filter_bleu_max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            # num_shards=cfg.distributed_training.distributed_world_size,
            # shard_id=cfg.distributed_training.distributed_rank,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
            # quick fix
            epoch=1,
            disable_iterator_cache=False,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            iterator,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"{rank_reprs} {self.__class__.__name__} Generate on {src}->{tgt} pair_dataset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        generator = task.build_generator(
            [model], cfg.generation, extra_gen_cls_kwargs={
                "symbols_to_strip_from_output": getattr(task, "symbols_to_strip", None)
            }
        )
        # Handle tokenization and BPE
        tokenizer = task.build_tokenizer(cfg.tokenizer)
        bpe = task.build_bpe(cfg.bpe)
        logger.info(f'{self.__class__.__name__}: {tokenizer=}, {bpe=}, {cfg.common_eval.post_process=}')

        def log_samples(_srcs, _tgts, _hyps, bpe=None):
            if bpe is None:
                bpe = lambda x: x
            for i in range(min(self.clu_cfg.aligner_log_n, len(_tgts))):
                logger.warning(f'{rank_reprs} [{i}] generate_synthetic_data gen-sync-data: \n'
                    f'\tsrc-{src}: {_srcs[i]}\n\ttgt-{tgt}: {_tgts[i]}\n\thyp-{tgt}: {_hyps[i]}')
                    
        def decode_fn(x):
            if bpe is not None:
                x = bpe.decode(x)
            if tokenizer is not None:
                x = tokenizer.decode(x)
            return x

        sample_ids = []
        eval_hypo_token_list = []
        eval_tgt_token_list = []
        src_str_list = []
        tgt_str_list = []
        detok_tgt_str_list = []
        detok_hyp_str_list = []
        logger.warning(f'{rank_reprs} Start generating....')
        acc_size = 0
        dummy_batch = None

        f_srcs, f_tgts, ex_srcs, ex_tgts = [], [], [], []
        for _step, sample in enumerate(progress):
            is_dummy = False
            if _step % 100 == 0:
                logger.warning(f'{rank_reprs} Generating {_step}, {len(f_srcs)=}, {len(ex_srcs)=}')
            if acc_size > cut_off:
                logger.warning(f'{rank_reprs} Stop generating due to cut_off {acc_size} / {cut_off}')
                break
            if sample is None or len(sample) == 0 or not bool(sample):
                logger.warning(f'{rank_reprs} Empty sample -> use dummy sample')
                assert dummy_batch is not None
                is_dummy = True
                sample = dummy_batch
            if dummy_batch is None:
                dummy_batch = sample

            # return sample
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample
            prefix_tokens = None
            hypos = task.inference_step(
                generator,
                [model],
                sample,
                prefix_tokens=prefix_tokens,
                constraints=None,
                bos_token=task.get_bos_token(tgt)
            )
            if is_dummy:
                continue
            assert "src_tokens" in sample['net_input']
            acc_size += sample['nsentences']

            for i, sample_id in enumerate(sample["id"].tolist()):
                # for j, hypo in enumerate(hypos[i][:cfg.generation.nbest]):
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], dictionary.pad())
                target_tokens = utils.strip_pad(
                    sample["target"][i, :], dictionary.pad()).int().cpu()
                raw_src_str = dictionary.string(src_tokens, 
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
                raw_tgt_str = dictionary.string(target_tokens,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))

                target_str = dictionary.string(
                    target_tokens,
                    cfg.common_eval.post_process,
                    escape_unk=True,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_target_str = decode_fn(target_str)

                hypo = hypos[i][0]
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=raw_src_str,
                    alignment=None,
                    align_dict=None,
                    tgt_dict=dictionary,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)

                eval_target_tokens = target_tokens.cpu()
                eval_hypo_tokens = hypo_tokens.cpu()
                if cfg.common_eval.post_process is not None:
                    # Convert back to tokens for evaluation with unk replacement and/or without BPE
                    eval_target_tokens = dictionary.encode_line(
                        detok_target_str, add_if_not_exist=True).cpu()
                    eval_hypo_tokens = dictionary.encode_line(
                        detok_hypo_str, add_if_not_exist=True).cpu()
                sample_ids.append(sample_id)
                eval_hypo_token_list.append(eval_hypo_tokens)
                eval_tgt_token_list.append(eval_target_tokens)
                src_str_list.append(raw_src_str)
                tgt_str_list.append(raw_tgt_str)
                detok_tgt_str_list.append(detok_target_str)
                detok_hyp_str_list.append(detok_hypo_str)

                if scorer is not None:
                    _score = scorer(eval_hypo_tokens, eval_target_tokens)
                    if filterer is not None:
                        f_src, f_tgt, ex_src, ex_tgt = filterer(raw_src_str, raw_tgt_str, _score)
                        f_srcs.extend(f_src)
                        f_tgts.extend(f_tgt)
                        ex_srcs.extend(ex_src)
                        ex_tgts.extend(ex_tgt)
                        if write_to_tmp and len(f_src) > 0:
                            src_file.writelines([x + '\n' for x in f_src])
                            tgt_file.writelines([x + '\n' for x in f_tgt])

        log_samples(src_str_list, detok_tgt_str_list, detok_hyp_str_list)
        if filterer is not None:
            return (sample_ids, eval_hypo_token_list, eval_tgt_token_list, src_str_list, tgt_str_list, 
                f_srcs, f_tgts, ex_srcs, ex_tgts)

        return sample_ids, eval_hypo_token_list, eval_tgt_token_list, src_str_list, tgt_str_list
    
    
    def generate_score_filter(self, model, task, pair_dataset, src, tgt, **kwargs):
        _dict = task.source_dictionary
        world_size, rank, rank_reprs = infer_dist_params(**kwargs)
        # compute synthetic -> score -> filter
        logger.warning(f'{self.__class__.__name__}{rank_reprs}: Start Generating synthetic dataset and filter')
        with SrcTgtMaybeWriter(
            src_path=kwargs.get('tmp_src_path', None),
            tgt_path=kwargs.get('tmp_tgt_path', None),
            write="tmp_src_path" in kwargs and "tmp_tgt_path" in kwargs
        ) as (src_file, tgt_file):
            (sample_ids, eval_hypo_token_list, eval_tgt_token_list,  src_str_list, 
                tgt_str_list, f_srcs, f_tgts, ex_srcs, ex_tgts) = self._generate_score_n_filter(
                    model, task, pair_dataset, src, tgt, 
                    scorer=lambda hyp, tgt: self.compute_sent_score(_dict, hyp, tgt),
                    filterer=lambda src_str, tgt_str, score: self.filter_sent_score(src_str, tgt_str, score),
                    src_file=src_file, 
                    tgt_file=tgt_file, 
                    **kwargs
            )
        # NOT do subsequently but continously because we may run into job requeue/OOM, so we still have some tmp data
        #   store, reactivate the follow lines if needed
        # scores = self.compute_scores(_dict, eval_hypo_token_list, eval_tgt_token_list)
        # f_srcs, f_tgts, ex_srcs, ex_tgts = self.filter_scores(src_str_list, tgt_str_list, scores)

        return f_srcs, f_tgts, ex_srcs, ex_tgts
    
    @torch.no_grad()
    def filter(self, data_pack, src, tgt, data_pack_as_path=False, return_exclude=False, overrides=None, **kwargs):
        cfg = self.cfg
        # model, task = self.load_eval_model_task()
        with self.load_eval_model_task(overrides=overrides) as (model, task):
            _dict = task.source_dictionary
            world_size, rank, rank_reprs = infer_dist_params(**kwargs)

            if data_pack_as_path:
                assert isinstance(data_pack, str)
                data_path = data_pack
                assert os.path.exists(data_path), f'{data_path} not found'
                assert hasattr(task, "load_translation_dataset")
                logger.info(f'{self.__class__.__name__}{rank_reprs} load binarized data at {data_path}, '
                    f'threshold={self.filter_bleu_threshold}, model={self.filter_model_path}'
                )
                pair_dataset = task.load_translation_dataset('train', data_path, pairs=[f'{src}-{tgt}'], combine=True)[0][1]
            else:
                assert isinstance(data_pack, dict)
                assert all(l in data_pack for l in self.langs), f'{self.langs} != {data_pack.keys()}'
                logger.info(f'{self.__class__.__name__}{rank_reprs} filtering {src}->{tgt}, '
                    f'threshold={self.filter_bleu_threshold}, model={self.filter_model_path}'
                )
                source_txts = data_pack[src]
                target_txts = data_pack[tgt]
                # convert data into respective dataset
                logger.warning(f'{self.__class__.__name__}{rank_reprs}: Start building inference dataset')
                _src_tokens = [_dict.encode_line(x, add_if_not_exist=False) for x in source_txts]
                _tgt_tokens = [_dict.encode_line(x, add_if_not_exist=False) for x in target_txts]
                _src_lengths = [len(x) for x in _src_tokens]
                _tgt_lengths = [len(x) for x in _tgt_tokens]
                pair_dataset = task.build_dataset_for_inference(
                    _src_tokens, _src_lengths, src, tgt=tgt, tgt_tokens=_tgt_tokens, tgt_lengths=_tgt_lengths)
            
            # compute synthetic -> score -> filter
            f_srcs, f_tgts, ex_srcs, ex_tgts = self.generate_score_filter(
                model, task, pair_dataset, src, tgt, **kwargs)

            out_pack = {src: f_srcs, tgt: f_tgts}
            exclude_pack = {src: ex_srcs, tgt: ex_tgts}
            total_size = len(f_srcs) + len(ex_srcs)
            logger.warning(f'{self.__class__.__name__}{rank_reprs}: Finish[{src}-{tgt}] => '
                f'filter {len(f_srcs)=} / exclude: {len(ex_srcs)=} '
                f'({len(f_srcs) / float(total_size) * 100}%)')

            del pair_dataset
            # NOTE: put in self.load_eval_model_task context to kill the model and task
        if return_exclude:
            return out_pack, exclude_pack
        return out_pack


class UMTBleuPercentileAlignedFilterer(UMTBleuAlignedFilterer):
    def __init__(self, cfg, langs) -> None:
        super().__init__(cfg, langs)
        self.filter_percentile = self.clu_cfg.filter_percentile
    
    def filter_scores(self, srcs, tgts, scores):
        # NOTE this can be hard values or percentile
        assert len(srcs) == len(tgts), f'{len(srcs)=} != {len(tgts)=}'
        assert len(srcs) == len(scores), f'{len(srcs)=} != {len(scores)=}'
        f_srcs = []
        f_tgts = []
        ex_srcs = []
        ex_tgts = []
        _start = int((1.0 - self.filter_percentile) * len(scores))
        sorted_indices = np.argsort(scores)[-_start:]
        for i, idx in enumerate(sorted_indices):
            _score = scores[idx]
            _src = srcs[idx]
            _tgt = tgts[idx]
            if self.filter_fullfil(_score):
                f_srcs.append(_src)
                f_tgts.append(_tgt)
            else:
                ex_srcs.append(_src)
                ex_tgts.append(_tgt)
        return f_srcs, f_tgts, ex_srcs, ex_tgts


class SimilarityPercentileAlignedFilterer(AlignedFilterer):
    def __init__(self, cfg, langs) -> None:
        super().__init__(cfg, langs)
        self.filter_percentile = self.clu_cfg.filter_percentile
    
    def filter(self, data_pack, src, tgt, data_pack_as_path=False, return_exclude=False, **kwargs):
        cfg = self.cfg
        sim_scores = kwargs.get('sim_scores')
        # NOTE: NEED TO CHECK if align_scores is similarity as ascending or descending
        #   similar ascending -> the larger the more similar
        #   NOTE: default similarity score: higher more similar topk-highest
        #   diffnorm:   lower the more similar (topk-lowest)
        #   cosine:     higher the more similar (topk-highes)

# ---------------
def build_clustering_algorithm(cfg, langs, **kwargs):
    mode = cfg.swav_clustering.clustering_mode
    if mode == 'prot_softmax_argmax':
        return SoftmaxArgmaxLangSepClusAlgo(cfg, langs)
    elif mode == 'prot_softmax':
        return SoftmaxSoftLangSepClusAlgo(cfg, langs)
    elif mode == 'kmeans':
        raise NotImplementedError(f'{mode=} not ready yet.')
    elif mode == 'pret_kmeans':
        raise NotImplementedError(f'{mode=} not ready yet.')
    else:
        raise ValueError(f'mode wrong {mode}')


def build_clustering_aligner(cfg, langs, clus_algo):
    aligner_name = cfg.swav_clustering.aligner
    if aligner_name == 'bipara_hard':
        return BiParaHardClusteringAligner(cfg, langs, clus_algo)
    elif aligner_name == 'diffnorm':
        # Diff norm is also L2 NOrm
        return DiffNormFasterBiParaClusteringAligner(cfg, langs, clus_algo)
    # elif aligner_name == 'diffnorm_p2p' or aligner_name == "diffnorm_br":
    #     return DiffNormDistBrBiParaClusteringAligner(cfg, langs, clus_algo)
    elif aligner_name in ['diffnorm_p2p', 'diffnorm_br', 'diffnorm_brfil']:
        return DiffNormFilDistBrBiParaClusteringAligner(cfg, langs, clus_algo)
    elif aligner_name == 'cosine':
        return CosineSimFasterBiParaClusteringAligner(cfg, langs, clus_algo)
    elif aligner_name in ['cosine_p2p', 'cosine_br', 'cosine_brfil']:
        return CosineSimFilDistBrBiParaClusteringAligner(cfg, langs, clus_algo)
    elif aligner_name == 'softmax_argmax':
        return SoftmaxArgmaxRankFasterBiParaCluAligner(cfg, langs, clus_algo)
    else:
        raise ValueError(f'aligner {aligner_name} invalid')


def build_clustering_filterer(cfg, langs, **kwargs):
    mode = cfg.swav_clustering.filter_mode
    logger.warning(f'Build filterer: {mode}')
    if mode is None or mode == 'default' or mode == "default_filter":
        return AlignedFilterer(cfg, langs)
    elif mode == 'umt_bleu':
        return UMTBleuAlignedFilterer(cfg, langs)
    elif mode == 'umt_bleu_percentile':
        return UMTBleuPercentileAlignedFilterer(cfg, langs)
    else:
        raise ValueError(f'filter_mode wrong {mode}')


# --- configure config
@dataclass
class AnalysisConfig(FairseqDataclass):
    analyze_name: Optional[str] = field(
        default="aly", metadata={"help": "path to lm checkpoint for lm fusion"},
    )
    analyze_max_step: int = field(
        default=500,
        metadata={
            "help": "ensures that every evaluated token has access to a context of at least this size, if possible"
        },
    )
    no_train_subset_shuffle: bool = field(
        default=False, metadata={"help": "default to shuffle train set"}
    )
    aly_para: bool = field(
        default=False, metadata={"help": "parallel data analysis, instead of mono data"}
    )
    no_aly_save: bool = field(
        default=False, metadata={"help": "save the data"}
    )
    aly_subsets: str = field(
        default="train", metadata={"help": "subset to run analysis"}
    )


@dataclass
class SwavClusteringConfig(FairseqDataclass):
    """
    Clustering gathering of pseudo parallel data for swav models
    """
    swav_langs: str = field(
        default=None, metadata={"help": "langs"},
    )
    # -- cluster algo settings
    clustering_mode: str = field(
        default="prot_softmax_argmax", metadata={"help": "Mode to sample clustering from prototypes output"},
    )
    kmeans_n_clusters: int = field(
        default=1000, metadata={"help": "no of clusters for kmeans computations"}
    )
    # -- cluster aligner settings -----------------------------------
    aligner: str = field(
        default="bipara_hard", metadata={"help": "Mode of aligner to obtain parallel data"},
    )
    cross_align_n: int = field(
        default=10, metadata={"help": "How many times to cross align sentences"}
    )
    cross_align_threshold: int = field(
        default=None, metadata={"help": "Threshold condition to sample sentences"}
    )
    cross_count_stop: int = field(
        default=-1, metadata={"help": "Stop aligning after specific amount of data"}
    )
    fwd_align_only: bool = field(
        default=False, metadata={"help": "Allow forward alignment only"}
    )
    aligner_log_n: int = field(
        default=10, metadata={"help": "Log n parallel samples"}
    )
    sim_fn: str = field(
        default="cosine", metadata={"help": "consine functions"},
    )
    without_replacement: bool = field(
        default=False, metadata={"help": "without_replacement, default is with replacement"}
    )
    # diff-norm mode
    diff_norm_mode: int = field(
        default=2, metadata={"help": "L1 or L2 norm for diff-norm mode"},
    )
    align_bsz: int = field(
        default=1000, metadata={"help": "Aligning batch size to do cuda"},
    )
    onspot_pre_filterer: str = field(
        default=None, metadata={"help": "On the spot filterer to accept/reject pair before computing score, "
            "it assign -inf score for rejected pairs"}
    )
    onspot_post_filterer: str = field(
        default=None, metadata={"help": "On the spot filterer to accept/reject pair after computing score, "
            "It filter scores after world, should be use in combination"}
    )
    onspot_margin_filterer: str = field(
        default=None, metadata={"help": "On the spot filterer to accept/reject pair after computing score, "
            "It filter scores right after post filterer, based on stats from the model"}
    )
    onspot_score_ninf: float = field(
        default=float('-inf'), metadata={"help": "similarity minimum scores"}
    )
    onspot_len_ratio: float = field(
        default=None, metadata={"help": "discard if len(x)/len(y) > len_ratio or < 1/len_ratio"}
    )
    onspot_threshold: float = field(
        default=None, metadata={"help": "Threshold to remove, only accept sim_score > threshold"}
    )
    onspot_margin_ratio: float = field(
        default=None, metadata={"help": "Threshold to remove, only accept sim_score > threshold"}
    )


    # filter -----------------------------------
    filter_only: bool = field(
        default=False, metadata={"help": "use previously created data set instead of generating new."}
    )
    filter_mode: str = field(
        default=None, metadata={"help": "Mode to Filter aligned data"},
    )
    filter_model_path: str = field(
        default=None, metadata={"help": "Path to filtering model"},
    )
    filter_model_task: str = field(
        default="translation", metadata={"help": "Task attached to the translation model"},
    )
    filter_bleu_threshold: float = field(
        default=10, metadata={"help": "Threshold to filter reference BLEU scores"}
    )
    filter_bleu_order: int = field(
        default=4, metadata={"help": "Threshold to filter reference BLEU scores"}
    )
    filter_bleu_max_tokens: float = field(
        default=512, metadata={"help": "Threshold to filter reference BLEU scores"}
    )
    filter_percentile: float = field(
        default=0, metadata={"help": "Percentile to keep, higher means stricter, more selective"}
    )
    filter_only_in_path: str = field(
        default=None, metadata={"help": "Path to dataset to filter FROM"},
    )
    filter_only_out_path: str = field(
        default=None, metadata={"help": "Path to dataset to filter INTO"},
    )

    # -- data export settings
    out_data_write_tmp: bool = field(
        default=False, metadata={"help": "Write aligned data to tmp text files"}
    )
    out_data_path: str = field(
        default=None, metadata={"help": "Path to data output"},
    )
    out_data_prefix: str = field(
        default="train", metadata={"help": "Path to data output"},
    )
    out_data_para_tag: str = field(
        default=None, metadata={"help": "src-tgt tag, e.g: train.en-ro.en.bin"},
    )
    out_data_workers: int = field(
        default=1, metadata={"help": "workers to preprocess"}
    )
    rm_exists: bool = field(
        default=False, metadata={"help": "Force remove data"}
    )

    # export -----------------------------------
    export_only: bool = field(
        default=False, metadata={"help": "Export into .pth data only"}
    )
    export_flush_steps: int = field(
        default=-1, metadata={"help": "Export Flush into multiple .pth data after x steps"}
    )


@dataclass
class ClusteringConfig(FairseqDataclass):
    cluster_algo: str = field(
        default="kmeans", metadata={"help": "algo use to train clustering"},
    )
    n_clusters: int = field(
        default=1000, metadata={"help": "inter cluster sampling data"}
    )
    cluster_name: str = field(
        default="kmeans", metadata={"help": "algo use to train clustering"},
    )


@dataclass
class GatherParaDataFairseqConfig(FairseqConfig):
    swav_clustering: SwavClusteringConfig = SwavClusteringConfig()
    analysis: AnalysisConfig = AnalysisConfig()
