# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.logging.meters import safe_round
from fairseq.criterions.wav2vec_criterion import Wav2VecCriterionConfig, Wav2vecCriterion
from typing import Callable, List, Optional, Sequence, Tuple
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
import math
from dataclasses import dataclass, field
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils, modules
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.distributed as dist
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.utils import is_xla_tensor, move_to_cuda

from fairseq.criterions.masked_lm import MaskedLmConfig, MaskedLmLoss

from ..swav_utils import check_num_stability, dist_cat, maybe_improve_stability, sinkhorn

logger = getLogger(__name__)


def build_std_swav_queue(
    rand_factor: int, 
    queue_length: int, 
    world_size: int, 
    embed: torch.Tensor, 
    no_rand_factor: bool = False, 
    with_langs: bool = False
):
    embed_dim = embed.size(1)
    if no_rand_factor:
        queue_d = {
            'queue': move_to_cuda(torch.zeros(queue_length // world_size, embed_dim).type_as(embed)),
            'qfill': move_to_cuda(torch.zeros(queue_length // world_size, dtype=torch.int8))
        }
        if with_langs:
            queue_d['langs'] = move_to_cuda(torch.Tensor(queue_length // world_size).long().fill_(-1))
    else:
        queue_d = {
            'queue': move_to_cuda(torch.zeros(rand_factor, queue_length // world_size, embed_dim).type_as(embed)),
            'qfill': move_to_cuda(torch.zeros(rand_factor, queue_length // world_size, dtype=torch.int8))
        }
        if with_langs:
            queue_d['langs'] = move_to_cuda(torch.Tensor(rand_factor, queue_length // world_size).long().fill_(-1))
    return queue_d


@dataclass
class StdSwavConfig(FairseqDataclass):
    rand_factor: int = field(
        default=2,
        metadata={"help": "random factor to multiple 1 to *rand_factor* samples"},
    )
    queue_length: int = field(
        default=0,
        metadata={"help": "queue length"},
    )
    swav_epsilon: float = field(
        default=0.05,
        metadata={"help": "swav epsilon"},
    )

    sinkhorn_iterations: int = field(
        default=3,
        metadata={"help": "sinkhorn no. iterations"},
    )

    epoch_queue_starts: int = field(
        default=-1,
        metadata={
            "help": "from this epoch, we start using a queue, not used for fairseq codebase,"
                    "use update_queue_starts instead"
        }
    )
    update_queue_starts: int = field(
        default=-1,
        metadata={"help": "from this update, start using the queue"}
    )

    swav_temperature: float = field(
        default=0.1,
        metadata={"help": "swav temperature"},
    )

    stability_epsilon: float = field(
        default=0.0,
        metadata={"help": "stability_epsilon to prevent NaN, recommend setting to 1e-8"},
    )
    # NOTE: this will reference distributed_training.distributed_world_size into the LossConfig
    distributed_world_size: int = II("distributed_training.distributed_world_size")
    pre_norm_prototypes: bool = field(
        default=False,
        metadata={"help": "Pre normalize prototypes before computing, need to turn this On!"},
    )

    improv_stab_global: bool = field(
        default=False,
        metadata={"help": "Improve stability by affective global in maybe_improve_stability"},
    )

    improve_numerical_stability: bool = field(
        default=True,
        metadata={"help": "improves numerical stability in Sinkhorn-Knopp algorithm"
                          "True, will store false if parse --improve-numerical-stability"},
    )


@dataclass
class ConstraintSwavConfig(StdSwavConfig):
    constraint_begin_up: int = field(
        default=-1,
        metadata={"help": "Only enable constraint after x updates"},
    )


class SwavCriterionWrapper(FairseqCriterion):
    """Use as MRO inheritance wraper for many kinds of swav loss attached to standard loss
    connect certain functions of swav loss:
    SwavCriterionWrapper should come first before base class
        and override the build_swav_loss functions to specify which swav loss to use
    E.g: SwavAndCrossEntropyLoss(SwavCriterionWrapper, CrossEntropyLoss):
        def build_swav_loss(self, **kwargs):
            return StdSwavCriterion(**kwargs)
    """
    def __init__(
        self, 
        task, 
        distributed_world_size,
        rand_factor,
        queue_length,
        swav_epsilon,
        sinkhorn_iterations,
        swav_temperature,
        stability_epsilon=0.0,
        pre_norm_prototypes=True,
        improve_numerical_stability=True,
        update_queue_starts=-1,
        improv_stab_global=False,
        **kwargs,
    ):
        logger.warning(f'{len(kwargs)}, {kwargs.keys()} {kwargs=}')
        kwargs['task'] = task
        super().__init__(**kwargs)
        self._swav = None
        self.build_swav_loss(
            distributed_world_size=distributed_world_size,
            rand_factor=rand_factor,
            queue_length=queue_length,
            swav_epsilon=swav_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            swav_temperature=swav_temperature,
            stability_epsilon=stability_epsilon,
            pre_norm_prototypes=pre_norm_prototypes,
            improve_numerical_stability=improve_numerical_stability,
            update_queue_starts=update_queue_starts,
            improv_stab_global=improv_stab_global,
        )
    
    def build_swav_loss(self, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def get_queue(self, *args, **kwargs):
        return self._swav.get_queue(*args, **kwargs)
    
    @torch.no_grad()
    def compute_sinkhorn_prototypes(self, *args, **kwargs):
        raise NotImplementedError
    
    def compute_swav_loss_out(self, *args, **kwargs):
        raise NotImplementedError
    
    def swav_sinkhorn_prototypes(self, *args, **kwargs):
        return self._swav._sinkhorn_prototypes(*args, **kwargs)
    
    def maybe_update_queue(self, *args, **kwargs):
        return self._swav.maybe_update_queue(*args, **kwargs)
    
    def compute_swav_loss(self, *args, **kwargs):
        return self._swav.compute_swav_loss(*args, **kwargs)


@register_criterion("std_swav", dataclass=StdSwavConfig)
class StdSwavCriterion(FairseqCriterion):
    """StdSwavCriterion impl vanilla swav loss functions
    How SWAV Loss is implemented:
    Further details in: https://arxiv.org/pdf/2006.09882.pdf
    0. Given input X
    1. Augment it into X1, X2, ..., Xn
    2. Pass them into encoder, obtain sent embeds z1, z2, ..., zn (also called prot_embed)
    3. Pass Z{z1,z2...} into prototype layer (C={c1, c2..., ck}) to obtain prototypes outputs (prot_output) p_i = z_i x C
    4. Pass prot_output into the sinkhorn algorithm, which produce probability-like scores Q={q1, q2, ..., qk}
        * note: sinkhorn algo performs something like softmax, whose output vectors sum up to 1
    5. Swap the codes Q of Z and apply weighted loss
        * loss_i = \sum_j (q_j * log_softmax(z_i x C) where j != i
            e.g: loss_{z2} = (q1 + q3 + q4....) * log_softmax(z2 x C)
    * For batched input, the arangement of data will be:
        [a1,b1,c1,d1,a2,b2,c2,d2] where a,b,c,d (bsz=4)are indivual samples Xa, Xb, Xc, Xd
                                        1,2 (rand_factor=2) are indices of different noised version of X
    """
    def __init__(
        self,
        task: FairseqTask,
        distributed_world_size: int,
        rand_factor: int,
        queue_length: int,
        swav_epsilon: float,
        sinkhorn_iterations: int,
        swav_temperature: float,
        stability_epsilon: float = 0.0,
        pre_norm_prototypes: bool = True,
        improve_numerical_stability: bool = True,
        update_queue_starts: int = -1,
        improv_stab_global: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            distributed_world_size: int: No. of all working GPUs (distributed), required for many distributed operations
            rand_factor: int: the number of times to augment X into -> X1, X2,..., X_{rand_factor}
            queue_length: int: the size of queue (total for entire process, will be divided to each gpu)
                               queue is accumulated and update with prot_embed (sent embeddings) as the training progress
                               When queue is full, sinkhorn output is computed over the queue, not just the current batch
            swav_epsilon: float: to be divided before .exp()
            sinkhorn_iterations: int: no. of iterations for sinkhorn algo
            swav_temperature: float: tmp to be divided by log_softmax(zC) when computing swav loss
            stability_epsilon: float = 0.0: to prevent certain instability issue
            pre_norm_prototypes: bool = True: pre-normalize prototype weight before computing loss
                                        (used by the swav_transformer model instead of in this loss)
            improve_numerical_stability: bool = True: improve stability by subtract maximum values of zC to prevent inf/nan
                                when computing .exp()
            update_queue_starts: int = -1: start using the queue after a number of updates
        """
        super().__init__(task, **kwargs)
        self.rand_factor = rand_factor
        self.queue_length = queue_length
        self.crops_for_assign = list(range(rand_factor))
        self.total_nmb_crops = len(self.crops_for_assign)
        self.swav_epsilon = swav_epsilon
        self.swav_temperature = swav_temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.swav_stability_epsilon = stability_epsilon
        self.improve_numerical_stability = improve_numerical_stability
        self.update_queue_starts = update_queue_starts
        self.distributed_world_size = distributed_world_size or torch.cuda.device_count()
        self.pre_norm_prototypes = pre_norm_prototypes
        self.improv_stab_global = improv_stab_global

        self._queue = None
        self.logsoftmax_layer = nn.LogSoftmax(dim=1)
        logger.info(
            f'{self.__class__.__name__}: randfac={self.rand_factor}, world={self.distributed_world_size},' +
            f'eps={self.swav_epsilon},temp={self.swav_temperature},sinkiter={self.sinkhorn_iterations},' +
            f'prenorm={self.pre_norm_prototypes},update_q_start{self.update_queue_starts},stab={self.swav_stability_epsilon}'
        )
    
    def get_queue(self, embed: torch.Tensor, no_rand_factor: bool = False, num_updates: Optional[int] = None, qname="default"):
        """Build the swav queue"""
        if embed is None:
            return None
        if num_updates is not None:
            if num_updates < self.update_queue_starts:
                return None
        if (self._queue is None or qname not in self._queue) and self.queue_length > 0:
            logger.warning(f'Create queue[{qname}]: {self.queue_length}, {num_updates=}')
            self._queue = {} if self._queue is None else self._queue
            self._queue[qname] = build_std_swav_queue(
                rand_factor=self.rand_factor, 
                queue_length=self.queue_length,
                world_size=self.distributed_world_size, 
                embed=embed,
                no_rand_factor=no_rand_factor
            )
        return self._queue[qname] if self._queue is not None else None

    def compute_sinkhorn_prototypes(self, model, sample, sample_key="net_input", queue=None, **kwargs):
        raise NotImplementedError('must be overwritten and use self._sinkhorn_prototypes')

    def compute_swav_loss_out(self, model, sample):
        raise NotImplementedError('To be impl at sub-class criterion')

    @torch.no_grad()
    def _sinkhorn_prototypes(
        self,
        prototypes: nn.Module,
        prot_out: torch.Tensor,
        embedding: torch.Tensor,
        bsz: int,
        queue: Optional[dict] = None
    ):
        """Compute sinkhorn prototypes codes q in eval mode, not training.
        """
        queue_d = queue
        if queue_d is not None:
            queue = queue_d["queue"]
            qfill = queue_d["qfill"]
        with torch.no_grad():
            out = prot_out.detach()
            out = out.float()
            # maybe_update_queue
            if queue_d is not None:
                last_row_filled = (qfill[-1] == 1).int()
                if last_row_filled == 1:
                    out = torch.cat([torch.mm(queue, prototypes.weight.t()), out], 0)
                queue[bsz:] = queue[:-bsz].clone()  # shift right
                queue[:bsz] = embedding[:min(bsz, queue.size(0))]
                qfill[bsz:] = qfill[:-bsz].clone()
                qfill[:bsz] = 1
            # out = self.maybe_update_queue(index, crop_id, bsz, queue_d, out, prototypes, embedding)
            q = out / self.swav_epsilon
            q = maybe_improve_stability(q, self.distributed_world_size, self.improve_numerical_stability, affect_global=self.improv_stab_global)
            q = torch.exp(q)
            check_num_stability('after exp', q)
            q = sinkhorn(q, self.sinkhorn_iterations, self.distributed_world_size, self.swav_stability_epsilon)[-bsz:]
        return q
    
    @torch.no_grad()
    def maybe_update_queue(
        self, 
        index: int, 
        crop_id: int, 
        bsz: int, 
        queue_d: Optional[dict], 
        out: torch.Tensor, 
        prototypes: nn.Module, 
        embedding: torch.Tensor
    ):
        """
        ============ swav loss ... ============
        1. Loop, each loop push embedding -> to queue until it is full (not torch.all(queue[i, -1, :] == 0)
        2. Once full, each loop out = concat([queue*prototypes, out] )
            but how to do if batch_size is variable?
            --> wait for all queues in all processes to be full!, then we can proceed
                each queue is updated with different frequency by different batch size
        """
        if queue_d is not None:
            queue = queue_d["queue"]
            qfill = queue_d["qfill"]
            last_row_filled = (qfill[index, -1] == 1).int()
            dist.all_reduce(last_row_filled, op=dist.ReduceOp.PRODUCT)
            if last_row_filled == 1:
                out = torch.cat(
                    [torch.mm(queue[index], prototypes.weight.t()), out], 0
                )
            # NOTE: bsz can be > queue_length, makesure per-gpu queue >-1024
            queue[index, bsz:] = queue[index, :-bsz].clone()  # shift right
            queue[index, :bsz] = embedding[
                crop_id * bsz:crop_id * bsz + min(bsz, queue[index].size(0))]  # fill first items
            qfill[index, bsz:] = qfill[index, :-bsz].clone()
            qfill[index, :bsz] = 1
        return out

    @torch.no_grad()
    def _training_sinkhorn_codes(
        self, 
        index: int, 
        crop_id: int, 
        bsz: int, 
        prototypes: nn.Module, 
        prot_out: torch.Tensor, 
        embedding: torch.Tensor, 
        queue_d: Optional[dict] = None, 
        presinkhorn_fn: Optional[Callable] = None, 
        postsinkhorn_fn: Optional[Callable] = None, 
        preexp_fn: Optional[Callable] = None, 
    ):
        """Compute the sinkhorn codes q = sinkhorn(z x C)
        Args:
            index: crop_id index
            crop_id: crop_id (noise version within range(rand_factor)), mostly index = crop_id
            bsz: actual batch size
            prototypes: the prototypes layer
            prot_out [bsz * rand_factor, nmb_prot]: prot output after prototypes layer
            embedding [bsz * rand_factor, dim]: sent embeddings before prottypes layer
            queue_d: the queue{'queue': torch.Tensor(queue_size, dim), 'qfill': torch.Tensor(queue_size)}
                storing info when the queue is used
            presinkhorn_fn: apply to q before going through sinkhorn algo (currently deactivated for std version)
            postsinkhorn_fn: apply to q after going through sinkhorn algo (currently deactivated for std version)
            preexp_fn: apply to q before running exp(q)
        """
        with torch.no_grad():
            out = prot_out[bsz * crop_id:bsz * (crop_id + 1)].detach()
            out = self.maybe_update_queue(index, crop_id, bsz, queue_d, out, prototypes, embedding)
            # NOTE: converting float help stability issues
            out = out.float()
            q = out / self.swav_epsilon
            q = maybe_improve_stability(q, self.distributed_world_size, self.improve_numerical_stability, affect_global=self.improv_stab_global)
            # q = preexp_fn(q)
            q = torch.exp(q)
            check_num_stability('after exp', q)
            assert self.improv_stab_global or torch.all(q.sum(-1) >= 1), f'{self.improv_stab_global}: after exp.sum < 1 {q.sum(-1)=}'
            # q = presinkhorn_fn(q)
            q = sinkhorn(q, self.sinkhorn_iterations, self.distributed_world_size, self.swav_stability_epsilon)[-bsz:]
            # q = postsinkhorn_fn(q)
        return q

    def compute_swav_loss_single(
        self, 
        prototypes: nn.Module, 
        prot_out: torch.Tensor, 
        embedding: torch.Tensor, 
        bsz: int, 
        queue: Optional[dict] = None,
    ):
        """Refer to class description for how swav loss works
        Args:
            prototypes nn.Module: the prototypes layer of the model
            prot_out torch.Tensor: the output after prototype layer
            embedding torch.Tensor: the output before prototype layer (encoder output, latent reprs)
            bsz int: the actual batch size, because the tensor batch size is inflated by rand_factor
                it is possible to infer bsz from prot_out.size(0) and self.rand_factor
                    bug it parsing bsz into the loss make sure this is the case.
        """
        
        loss = 0
        queue_d = queue
        for i, crop_id in enumerate(self.crops_for_assign):
            q = self._training_sinkhorn_codes(
                i, crop_id, bsz, prototypes, prot_out, embedding, queue_d)
            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(self.total_nmb_crops), crop_id):
                log_probs = self.logsoftmax_layer(
                    prot_out[bsz * v:bsz * (v + 1)] / self.swav_temperature
                )
                subloss -= torch.mean(torch.sum(q * log_probs, dim=1))
                # check_num_stability('subloss', subloss)
            loss += subloss / (self.total_nmb_crops - 1)
        loss /= len(self.crops_for_assign)
        return loss
    
    def compute_swav_loss(
        self, 
        prototypes: nn.Module, 
        prot_out: torch.Tensor, 
        embedding: torch.Tensor, 
        bsz: int, 
        queue: Optional[dict] = None,
        sec_prot_out: Optional[torch.Tensor] = None,
        sec_embedding: Optional[torch.Tensor] = None,
        sec_queue: Optional[dict] = None,
    ):
        """Refer to class description for how swav loss works
        Args:
            prototypes nn.Module: the prototypes layer of the model
            prot_out torch.Tensor: the output after prototype layer
            embedding torch.Tensor: the output before prototype layer (encoder output, latent reprs)
            bsz int: the actual batch size, because the tensor batch size is inflated by rand_factor
                it is possible to infer bsz from prot_out.size(0) and self.rand_factor
                    bug it parsing bsz into the loss make sure this is the case.
        """
        
        loss = 0
        queue_d = queue
        sec_queue_d = sec_queue
        for i, crop_id in enumerate(self.crops_for_assign):
            q = self._training_sinkhorn_codes(
                i, crop_id, bsz, prototypes, prot_out, embedding, queue_d)
            
            codes_q = [q]
            prots = [prot_out]
            if sec_prot_out is not None:
                q_sec = self._training_sinkhorn_codes(
                    i, crop_id, bsz, prototypes, sec_prot_out, sec_embedding, sec_queue_d)
                codes_q.append(q_sec)
                prots.append(sec_prot_out)
            # cluster assignment prediction
            subloss = 0
            loop_counts = 0
            for j, _q in enumerate(codes_q):
                for k, _prot in enumerate(prots):
                    for v in np.delete(np.arange(self.total_nmb_crops), crop_id):
                        log_probs = self.logsoftmax_layer(
                            _prot[bsz * v:bsz * (v + 1)] / self.swav_temperature
                        )
                        subloss -= torch.mean(torch.sum(_q * log_probs, dim=1))
                    loss += subloss / (self.total_nmb_crops - 1)
                    loop_counts += 1
            loss /= loop_counts
        loss /= len(self.crops_for_assign)
        return loss


@register_criterion("langagnostic_c1_swav", dataclass=ConstraintSwavConfig)
class LangAgnosticConst1SwavCriterion(StdSwavCriterion):
    """Version 1 of Language-agnostic constraint for Swav loss
    How vanilla SWAV loss works: refer to StdSwavCriterion
    How the constraint works is described in compute_mulling_q_cluster_probs function
    """
    
    def get_queue(
        self, 
        embed: torch.Tensor, 
        no_rand_factor: bool = False, 
        num_updates: Optional[int] = None, 
        qname='default'
    ):
        if embed is None:
            return None
        if num_updates is not None:
            if num_updates < self.update_queue_starts:
                return None
        if (self._queue is None or qname not in self._queue) and self.queue_length > 0:
            # FIXME nxphi: need to specify how many updates to start using queue
            logger.warning(f'Create queue: len={self.queue_length}, with_langs, {num_updates=}')
            self._queue = {} if self._queue is None else self._queue
            self._queue[qname] = build_std_swav_queue(
                rand_factor=self.rand_factor, queue_length=self.queue_length,
                world_size=self.distributed_world_size, embed=embed,
                no_rand_factor=no_rand_factor,
                with_langs=True
            )
        return self._queue[qname] if self._queue is not None else None
    
    @torch.no_grad()
    def _sinkhorn_prototypes(self, prototypes, prot_out, embedding, bsz, langs, queue=None):
        queue_d = queue
        if queue_d is not None:
            queue = queue_d["queue"]
            qfill = queue_d["qfill"]
            qlangs = queue_d["langs"]
        with torch.no_grad():
            out = prot_out.detach()
            out_langs = langs
            out = out.float()
            # maybe_update_queue
            if queue_d is not None:
                last_row_filled = (qfill[-1] == 1).int()
                if last_row_filled == 1:
                    out = torch.cat([torch.mm(queue, prototypes.weight.t()), out], 0)
                    out_langs = torch.cat((qlangs, out_langs), 0)
                queue[bsz:] = queue[:-bsz].clone()  # shift right
                queue[:bsz] = embedding[:min(bsz, queue.size(0))]
                qlangs[bsz:] = qlangs[:-bsz].clone()  # shift right
                qlangs[:bsz] = langs[:min(bsz, queue.size(0))]
                qfill[bsz:] = qfill[:-bsz].clone()
                qfill[:bsz] = 1
            # out = self.maybe_update_queue(index, crop_id, bsz, queue_d, out, prototypes, embedding)
            q = out / self.swav_epsilon
            q = maybe_improve_stability(q, self.distributed_world_size, self.improve_numerical_stability, affect_global=self.improv_stab_global)
            q = torch.exp(q)
            check_num_stability('after exp', q)
            q = sinkhorn(q, self.sinkhorn_iterations, self.distributed_world_size, self.swav_stability_epsilon)[-bsz:]
        return q
    
    @torch.no_grad()
    def maybe_update_queue(self, index, crop_id, bsz, queue_d, out, out_langs, langs, prototypes, embedding):
        if queue_d is not None:
            assert "langs" in queue_d, f'langs must be in queue'
            queue = queue_d["queue"]
            qfill = queue_d["qfill"]
            qlangs = queue_d["langs"]
            last_row_filled = (qfill[index, -1] == 1).int()
            dist.all_reduce(last_row_filled, op=dist.ReduceOp.PRODUCT)
            if last_row_filled == 1:
                out = torch.cat(
                    [torch.mm(queue[index], prototypes.weight.t()), out], 0
                )
                out_langs = torch.cat((qlangs[index], out_langs), 0)
            # if bsz > queue[index].size(0):
            #     logger.warning(f'bsz[{bsz}] > queue [{queue[index].size()}], increase queue-length or it behave wrongly')
            queue[index, bsz:] = queue[index, :-bsz].clone()  # shift right
            queue[index, :bsz] = embedding[
                crop_id * bsz:crop_id * bsz + min(bsz, queue[index].size(0))]   # fill 1st
            qlangs[index, bsz:] = qlangs[index, :-bsz].clone()
            qlangs[index, :bsz] = langs[
                crop_id * bsz:crop_id * bsz + min(bsz, queue[index].size(0))]
            qfill[index, bsz:] = qfill[index, :-bsz].clone()
            qfill[index, :bsz] = 1
        return out, out_langs

    @torch.no_grad()
    def _training_sinkhorn_codes(
        self, 
        index: int, 
        crop_id: int, 
        bsz: int, prototypes: nn.Module, 
        prot_out: torch.Tensor, 
        embedding: torch.Tensor, 
        langs: torch.Tensor, 
        queue_d: Optional[dict] = None, 
        presinkhorn_fn: Optional[Callable] = None, 
        postsinkhorn_fn: Optional[Callable] = None, 
        preexp_fn: Optional[Callable] = None, 
        truncate_queue: bool = True
    ):
        presinkhorn_fn = (lambda x: x) if presinkhorn_fn is None else presinkhorn_fn
        postsinkhorn_fn = (lambda x: x) if postsinkhorn_fn is None else postsinkhorn_fn
        preexp_fn = (lambda x: x) if preexp_fn is None else preexp_fn

        out = prot_out[bsz * crop_id:bsz * (crop_id + 1)].detach()
        out_langs = langs[bsz * crop_id:bsz * (crop_id + 1)].detach()
        # time to use the queue
        out, out_langs = self.maybe_update_queue(
            index, crop_id, bsz, queue_d, out, out_langs, langs, prototypes, embedding
        )
        # NOTE: converting float help stability issues
        out = out.float()

        q = out / self.swav_epsilon
        q = maybe_improve_stability(q, self.distributed_world_size, self.improve_numerical_stability, affect_global=self.improv_stab_global)
        q = preexp_fn(q)
        q = torch.exp(q)
        check_num_stability('after exp', q)
        assert self.improv_stab_global or torch.all(q.sum(-1) >= 1), f'{self.improv_stab_global}: after exp.sum < 1 {q.sum(-1)=}'
        q = presinkhorn_fn(q)
        q = sinkhorn(q, self.sinkhorn_iterations, self.distributed_world_size, self.swav_stability_epsilon)
        if truncate_queue:
            q = q[-bsz:]
            out_langs = out_langs[-bsz:]
        q = postsinkhorn_fn(q)
        return q, out_langs

    @torch.no_grad()
    def compute_mulling_q_cluster_probs(
        self, 
        bsz: int, 
        std_crops_q: List[torch.Tensor], 
        std_crops_langs: List[torch.Tensor], 
        dist_lang_ids: torch.Tensor, 
        langs: Optional[torch.Tensor] = None, 
    ):
        """Cluster have to create mask for q of each crop
        * v1: negative per-lang average score
            1. compute distributed lang_q = q[(bsz+queue)*world, dim] for each lang
            2. compute per-lang average langq_scores[dim] of lang_q
            3. determine how the per-lang weight would be base on lang_q_scores
            4. langq_score = 1.0 - (lang_avg_q / sum(all lang_avg_q))
            e.g: lang_q_all: [[0.1, 0.5, 0.4], -> 1.0 - [[0.33, 0.83, 0.36], -> [[0.66, 0.17, 0.63],
                              [0.2, 0.1, 0.7]]           [0.66, 0.16, 0.63]      [0.33, 0.84, 0.34]
            -- v1 does not take into account the relative strength of different classes, it only tries 
            to balance between languages, not specifically spread evenly the class assignment distribution
        -->
        args:
            bsz: original batch size
            std_crops_q:        [tensor[bsz + queue, dim], ...]
            std_crops_langs:    [tensor[bsz + queue], ...]
        return:
            q_cluster_masks: [mask[bsz + queue, dim], ...]
            masks: [n_langs, 1, d]
        """
        if len(dist_lang_ids) == 1:
            return None, None
        assert len(std_crops_q) == len(std_crops_langs), f'{len(std_crops_q)=} != {len(std_crops_langs)=}'
        assert len(std_crops_q) == len(self.crops_for_assign)
        assert all(
            x.size(0) == y.size(0) for x, y in zip(std_crops_q, std_crops_langs)
        ), f'{std_crops_q=}/\n{std_crops_langs=}'
            
        lang_all_avg_q = []
        dist_lang_ids = dist_lang_ids.tolist()
        for lidx, lang_id in enumerate(dist_lang_ids):
            lang_cql = std_crops_q[0].new(std_crops_q[0].size(-1)).fill_(0)
            lang_cql_count = std_crops_q[0].new(1).int().fill_(0)
            for j, crop_id in enumerate(self.crops_for_assign):
                cq = std_crops_q[j]
                cl = std_crops_langs[j]
                cql = cq[cl == lang_id]
                lang_cql += cql.sum(0)
                lang_cql_count += cql.size(0)
            dist.all_reduce(lang_cql, op=dist.ReduceOp.SUM)
            dist.all_reduce(lang_cql_count, op=dist.ReduceOp.SUM)
            lang_all_avg_q.append((lang_cql / lang_cql_count).unsqueeze_(0))
        lang_all_avg_q = torch.cat(lang_all_avg_q, 0)
        lang_q_score = 1.0 - (lang_all_avg_q / (lang_all_avg_q.sum(0, keepdim=True) + 1e-6))
        masks = lang_q_score.unsqueeze(1)
        # lang_all_avg_q:   [n_langs, d]
        # masks :           [n_langs, 1, d]
        q_cluster_masks = []
        for i, crop_id in enumerate(self.crops_for_assign):
            q_mask = std_crops_q[0].new(std_crops_q[0].size()).fill_(-1)
            # q_mask:       [bsz+queue, d]
            for lidx, lang_id in enumerate(dist_lang_ids):
                lang_mask = (std_crops_langs[crop_id] == lang_id).unsqueeze_(-1)
                # lang_mask: [bsz+queue, 1]
                q_mask = q_mask.masked_scatter_(lang_mask, masks[lidx].expand_as(q_mask))
            q_cluster_masks.append(q_mask)
        return q_cluster_masks, masks

    def build_presinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        if is_lang_agnos:
            def presinkhorn_fn(x):
                # x is after exp(x): before pre-sinkhorn
                assert x.size(0) == q_cluster_masks[index].size(0), f'[{index}]: {x.size()=}!={q_cluster_masks[index]}'
                return x * q_cluster_masks[index]
        else:
            def presinkhorn_fn(x):
                return x
        return presinkhorn_fn

    def build_postsinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None

    def build_preexp_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None

    def compute_swav_loss(
        self, 
        prototypes: nn.Module, 
        prot_out: torch.Tensor, 
        embedding: torch.Tensor, 
        bsz: int, 
        langs: torch.Tensor, 
        queue: Optional[dict] = None, 
        sec_prot_out: Optional[torch.Tensor] = None,
        sec_embedding: Optional[torch.Tensor] = None,
        sec_queue: Optional[dict] = None,
        no_constraint=False,
    ):
        """
        * langs is required, langs is also needed in queue, if queue is not None
        langs: [bsz,]
        if no_constraint -> force no lang-ag constraint, incase we want to dynamically control contraints.
            control by constraint_begin_up variables
        """
        # TODO(nxphi): constraint_begin_up , control whether to use constraint at specific updates
        unique_lang_ids = langs.unique().cpu()
        assert len(unique_lang_ids) > 0, 'unique_lang_ids: {}'.format(unique_lang_ids)
        dist_lang_ids = torch.unique(dist_cat(self.distributed_world_size, unique_lang_ids, 0))
        is_lang_agnos = (len(dist_lang_ids) > 1) and not no_constraint
        q_cluster_masks, q_masks = None, None
        
        # multiple languages are available, compute prototypes codes
        queue_d = queue
        sec_queue_d = sec_queue
        if is_lang_agnos:
            # ======= compute standard prototypes codes =========
            std_crops_q = []
            std_crops_langs = []
            sec_std_crops_q = []
            sec_std_crops_langs = []

            with torch.no_grad():
                queue_d_proc = {k: v.clone() for k, v in queue_d.items()} if queue_d is not None else None
                sec_queue_d_proc = {k: v.clone() for k, v in sec_queue_d.items()} if sec_queue_d is not None else None
                for i, crop_id in enumerate(self.crops_for_assign):
                    q_std, q_out_langs = self._training_sinkhorn_codes(
                        i, crop_id, bsz, prototypes, prot_out, embedding, langs, queue_d_proc, truncate_queue=False)
                    std_crops_q.append(q_std)
                    std_crops_langs.append(q_out_langs)
                    if sec_prot_out is not None:
                        sec_q_std, sec_q_out_langs = self._training_sinkhorn_codes(
                            i, crop_id, bsz, prototypes, sec_prot_out, sec_embedding, langs, sec_queue_d_proc, truncate_queue=False)
                        sec_std_crops_q.append(sec_q_std)
                        sec_std_crops_langs.append(sec_q_out_langs)

            # ======= compute clusters mask pre/post-sinkhorn fn =========
            del queue_d_proc, sec_queue_d_proc
            q_cluster_masks, q_masks = self.compute_mulling_q_cluster_probs(
                bsz, std_crops_q, std_crops_langs, dist_lang_ids, langs
            )
            if sec_prot_out is not None:
                sec_q_cluster_masks, sec_q_masks = self.compute_mulling_q_cluster_probs(
                    bsz, sec_std_crops_q, sec_std_crops_langs, dist_lang_ids, langs
                )
            del std_crops_q, std_crops_langs, sec_std_crops_q, sec_std_crops_langs
        # ======= compute final swav loss with modified codes ==========
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            q, out_langs = self._training_sinkhorn_codes(
                i, crop_id, bsz, prototypes, prot_out, embedding, langs, queue_d, 
                presinkhorn_fn=self.build_presinkhorn_fn(i, crop_id, is_lang_agnos, q_cluster_masks, q_masks),
                postsinkhorn_fn=self.build_postsinkhorn_fn(i, crop_id, is_lang_agnos, q_cluster_masks, q_masks),
                preexp_fn=self.build_preexp_fn(i, crop_id, is_lang_agnos, q_cluster_masks, q_masks),
            )
            codes_q = [q]
            prots = [prot_out]
            if sec_prot_out is not None:
                sec_q, sec_out_langs = self._training_sinkhorn_codes(
                    i, crop_id, bsz, prototypes, sec_prot_out, sec_embedding, langs, sec_queue_d, 
                    presinkhorn_fn=self.build_presinkhorn_fn(i, crop_id, is_lang_agnos, sec_q_cluster_masks, sec_q_masks),
                    postsinkhorn_fn=self.build_postsinkhorn_fn(i, crop_id, is_lang_agnos, sec_q_cluster_masks, sec_q_masks),
                    preexp_fn=self.build_preexp_fn(i, crop_id, is_lang_agnos, sec_q_cluster_masks, sec_q_masks),
                )
                codes_q.append(sec_q)
                prots.append(sec_prot_out)

            # cluster assignment prediction
            subloss = 0
            loop_counts = 0
            for j, _q in enumerate(codes_q):
                for k, _prot in enumerate(prots):
                    for v in np.delete(np.arange(self.total_nmb_crops), crop_id):
                        log_probs = self.logsoftmax_layer(
                            _prot[bsz * v:bsz * (v + 1)] / self.swav_temperature
                        )
                        subloss -= torch.mean(torch.sum(_q * log_probs, dim=1))
                    loss += subloss / (self.total_nmb_crops - 1)
                    loop_counts += 1
            loss /= loop_counts
        loss /= len(self.crops_for_assign)
        return loss


@register_criterion("edlangagnostic_c1_swav", dataclass=ConstraintSwavConfig)
class EncDecLangAgnosticConst1SwavCriterion(LangAgnosticConst1SwavCriterion):
    pass


@register_criterion("langagnostic_c1_preexp_swav", dataclass=ConstraintSwavConfig)
class LangAgnosticConst1PreExpSwavCriterion(LangAgnosticConst1SwavCriterion):
    def build_presinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None
    
    def build_postsinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None
    
    def build_preexp_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        if is_lang_agnos:
            def preexp_fn(x):
                # x is after exp(x)
                assert x.size(0) == q_cluster_masks[index].size(0), f'[{index}]: {x.size()=}!={q_cluster_masks[index]}'
                return x * q_cluster_masks[index]
        else:
            def preexp_fn(x):
                return x
        return preexp_fn


@register_criterion("langagnostic_c2_swav", dataclass=ConstraintSwavConfig)
class LangAgnosticConst2SwavCriterion(LangAgnosticConst1SwavCriterion):
    @torch.no_grad()
    def compute_mulling_q_cluster_probs(
        self, 
        bsz: int, 
        std_crops_q: List[torch.Tensor], 
        std_crops_langs: List[torch.Tensor], 
        dist_lang_ids: torch.Tensor, 
        langs: Optional[torch.Tensor] = None
    ):
        """Cluster have to create mask for q of each crop
        * v2: negative score
            1. compute distributed lang_q = q[(bsz+queue)*world, dim] for each lang
            2. compute per-lang average langq_scores[dim] of lang_q
            3. determine how the per-lang weight would be base on lang_q_scores
            4. langq_score = 1.0 - (lang_avg_q)
            e.g: lang_q_all: 1 - [[0.1, 0.5, 0.4], -> [[0.9, 0.5, 0.6],
                                  [0.2, 0.1, 0.7]]     [0.8, 0.9, 0.3]
            -- v1 does not take into account the relative strength of different classes, it only tries 
            to balance between languages, not specifically spread evenly the class assignment distribution
        -->
        args:
            bsz: original batch size
            std_crops_q:        [tensor[bsz + queue, dim], ...]
            std_crops_langs:    [tensor[bsz + queue], ...]
        return:
            q_cluster_masks: [mask[bsz + queue, dim], ...]
            masks: [n_langs, 1, d]
        """
        if len(dist_lang_ids) == 1:
            return None, None
        assert len(std_crops_q) == len(std_crops_langs), f'{len(std_crops_q)=} != {len(std_crops_langs)=}'
        assert len(std_crops_q) == len(self.crops_for_assign)
        assert all(
            x.size(0) == y.size(0) for x, y in zip(std_crops_q, std_crops_langs)
        ), f'{std_crops_q=}/\n{std_crops_langs=}'

        lang_all_avg_q = []
        for lid, lang_id in enumerate(dist_lang_ids):
            lang_cql = std_crops_q[0].new(std_crops_q[0].size(-1)).fill_(0)
            lang_cql_count = std_crops_q[0].new(1).long().fill_(0)
            for j, crop_id in enumerate(self.crops_for_assign):
                cq = std_crops_q[j]
                cl = std_crops_langs[j]
                assert cq.size(0) == cl.size(0), '{} != {}'.format(cq.size(), cl.size())
                cql = cq[cl == lang_id]
                lang_cql += cql.sum(0)
                lang_cql_count += cql.size(0)
            dist.all_reduce(lang_cql, op=dist.ReduceOp.SUM)
            dist.all_reduce(lang_cql_count, op=dist.ReduceOp.SUM)
            lang_all_avg_q.append((lang_cql / lang_cql_count).unsqueeze_(0))
        lang_all_avg_q = torch.cat(lang_all_avg_q, 0)
        # lang_all_avg_q:   [n_langs, d]
        lang_q_score = 1.0 - (lang_all_avg_q)
        masks = lang_q_score.unsqueeze(1)
        # masks :           [n_langs, 1, d]
        q_cluster_masks = []
        for i, crop_id in enumerate(self.crops_for_assign):
            q_mask = std_crops_q[0].new(std_crops_q[0].size()).fill_(-1)
            # q_mask:       [bsz+queue, d]
            for lidx, lang_id in enumerate(dist_lang_ids):
                lang_mask = (std_crops_langs[crop_id] == lang_id).unsqueeze_(-1)
                # lang_mask: [bsz+queue, 1]
                assert q_mask.size(0) == lang_mask.size(0), '{} != {}'.format(q_mask.size(), lang_mask.size())
                q_mask = q_mask.masked_scatter_(lang_mask, masks[lidx].expand_as(q_mask))
            q_cluster_masks.append(q_mask)
        return q_cluster_masks, masks
    
    def build_presinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        if is_lang_agnos:
            def presinkhorn_fn(x):
                # x is after exp(x)
                assert x.size(0) == q_cluster_masks[index].size(0), f'[{index}]: {x.size()=}!={q_cluster_masks[index]}'
                return x * q_cluster_masks[index]
        else:
            def presinkhorn_fn(x):
                return x
        return presinkhorn_fn
    
    def build_postsinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None
    
    def build_preexp_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None


@register_criterion("langagnostic_c3_swav", dataclass=ConstraintSwavConfig)
class LangAgnosticConst3SwavCriterion(LangAgnosticConst1SwavCriterion):
    @torch.no_grad()
    def compute_mulling_q_cluster_probs(self, 
        bsz: int, 
        std_crops_q: List[torch.Tensor], 
        std_crops_langs: List[torch.Tensor], 
        dist_lang_ids: torch.Tensor, 
        langs: Optional[torch.Tensor] = None
    ):
        """Cluster have to create mask for q of each crop
        * v3: Inverse assignment frequency bump
            1. compute distributed lang_q = q[(bsz+queue)*world, dim] for each lang
            2. convert lang_q -> one-hot Q_x^{1h}
            3. Count per-cluster assignment $F_x = Q_x^{1h}.sum(0)$
            4. $w_{x} = 1.0 - (F_x / (\sum F_x))$
            ***** HOWEVER, this may cause the scores to be very low!
        args:
            bsz: original batch size
            std_crops_q:        [tensor[bsz + queue, dim], ...]
            std_crops_langs:    [tensor[bsz + queue], ...]
        return:
            q_cluster_masks: [mask[bsz + queue, dim], ...]
            masks: [n_langs, 1, d]
        """
        if len(dist_lang_ids) == 1:
            return None, None
        assert len(std_crops_q) == len(std_crops_langs), f'{len(std_crops_q)=} != {len(std_crops_langs)=}'
        assert len(std_crops_q) == len(self.crops_for_assign)
        assert all(x.size(0) == y.size(0) for x, y in zip(std_crops_q, std_crops_langs))

        # eps = self.swav_stability_epsilon
        n_classes = std_crops_q[0].size(-1)
        lang_all_q_freq = []
        for lid, lang_id in enumerate(dist_lang_ids):
            lang_q_1h_sum = std_crops_q[0].new(std_crops_q[0].size(-1)).long().fill_(0)
            for j, crop_id in enumerate(self.crops_for_assign):
                cq = std_crops_q[j]
                cl = std_crops_langs[j]
                cql = cq[cl == lang_id]
                if cql.size(0) > 0:
                    cql_1h = F.one_hot(cql.argmax(-1), num_classes=n_classes)
                    lang_q_1h_sum += cql_1h.sum(0)
            dist.all_reduce(lang_q_1h_sum, op=dist.ReduceOp.SUM)
            lang_all_q_freq.append(lang_q_1h_sum.unsqueeze_(0))
        lang_all_q_freq = torch.cat(lang_all_q_freq, 0)
        lang_q_score = 1.0 - (lang_all_q_freq / (lang_all_q_freq.sum(1, keepdim=True) + 1e-6))
        masks = lang_q_score.unsqueeze(1)
        # lang_all_avg_q:   [n_langs, d]
        # lang_q_scoare:    [n_langs, d]
        # masks :           [n_langs, 1, d]
        q_cluster_masks = []
        for i, crop_id in enumerate(self.crops_for_assign):
            q_mask = std_crops_q[0].new(std_crops_q[0].size()).fill_(-1)
            # q_mask:       [bsz+queue, d]
            for lidx, lang_id in enumerate(dist_lang_ids):
                lang_mask = (std_crops_langs[crop_id] == lang_id).unsqueeze_(-1)
                # lang_mask: [bsz+queue, 1]
                q_mask = q_mask.masked_scatter_(lang_mask, masks[lidx].expand_as(q_mask))
            q_cluster_masks.append(q_mask)
        return q_cluster_masks, masks
    
    def build_presinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        if is_lang_agnos:
            def presinkhorn_fn(x):
                # x is after exp(x)
                assert x.size(0) == q_cluster_masks[index].size(0), f'[{index}]: {x.size()=}!={q_cluster_masks[index]}'
                return x * q_cluster_masks[index]
        else:
            presinkhorn_fn = lambda x: x
        return presinkhorn_fn
    
    def build_postsinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None
    
    def build_preexp_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None


@register_criterion("langagnostic_c3_preexp_swav", dataclass=ConstraintSwavConfig)
class LangAgnosticConst3PreExpSwavCriterion(LangAgnosticConst3SwavCriterion):
    def build_presinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None
    
    def build_postsinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None
    
    def build_preexp_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        if is_lang_agnos:
            def preexp_fn(x):
                assert x.size(0) == q_cluster_masks[index].size(0), f'[{index}]: {x.size()=}!={q_cluster_masks[index]}'
                # x must be >= 0, other wise the mask weight wont' behave correctly
                assert torch.all(x >= 0), f'[{index}] x not all >=0: {x[x < 0]}'
                return x * q_cluster_masks[index]
        else:
            preexp_fn = lambda x: x
        return preexp_fn


@register_criterion("langagnostic_c3_preexpplus_swav", dataclass=ConstraintSwavConfig)
class LangAgnosticConst3PreExpPlusSwavCriterion(LangAgnosticConst3SwavCriterion):
    def build_presinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None
    
    def build_postsinkhorn_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        return None
    
    def build_preexp_fn(self, index, crop_id, is_lang_agnos, q_cluster_masks, q_masks):
        if is_lang_agnos:
            def preexp_fn(x):
                assert x.size(0) == q_cluster_masks[index].size(0), f'[{index}]: {x.size()=}!={q_cluster_masks[index]}'
                # x must be >= 0, other wise the mask weight wont' behave correctly
                return x + x.abs() * q_cluster_masks[index]
        else:
            preexp_fn = lambda x: x
        return preexp_fn


@register_criterion("freq_langag_swav", dataclass=ConstraintSwavConfig)
class FreqLangAgnosticConstSwavCriterion(LangAgnosticConst3PreExpPlusSwavCriterion):
    pass


class PiecewiseLinearFn:
    """Piecewise linear function. Can be configured with a string."""

    def __init__(self, pieces: Sequence[Tuple[int, float]]):
        assert pieces == sorted(
            pieces
        ), f"PiecewiseLinearFn configuration should be sorted, received: {pieces}"

        self.pieces = pieces

    def __call__(self, x: int) -> float:
        for i, (x_a, y_a) in enumerate(self.pieces[:-1]):
            x_b, y_b = self.pieces[i + 1]
            if x_a <= x <= x_b:
                return y_a + (x - x_a) * (y_b - y_a) / (x_b - x_a)

        return self.pieces[-1][1]

    @staticmethod
    def from_string(configuration: str) -> "PiecewiseLinearFn":
        """
        Parse the configuration of lambda coefficient (for scheduling).
        x = "3"                  # lambda will be a constant equal to x
        x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                                 # to 0 during the first 1000 iterations
        x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                                 # iterations, then will linearly increase to 1 until iteration 2000
        """
        if isinstance(configuration, float):
            return PiecewiseLinearFn([(0, configuration)])

        try:
            parts = configuration.split(",")
            if len(parts) == 1:
                v = float(configuration)
                return PiecewiseLinearFn([(0, v)])

            split = [s.split(":") for s in parts]
            pieces = [(int(t), float(v)) for t, v in split]
            return PiecewiseLinearFn(pieces)
        except Exception:
            raise ValueError(
                f"Invalid PiecewiseLinearFn configuration: {configuration!r}"
            )

    @staticmethod
    def one() -> "PiecewiseLinearFn":
        return PiecewiseLinearFn([(0, 1.0)])


@dataclass
class SwavLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig, StdSwavConfig):
    swav_lambda: str = field(
        default="0.5",
        metadata={"help": "swav lambda, if < 1: loss=(1.0 - lambda)lm + lambda*swav, if > 1: loss=lm + (lambda - 1)*swav"},
    )


@register_criterion(
    "swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig)
class SwavLabelSmoothedCrossEntropyCriterion(SwavCriterionWrapper, LabelSmoothedCrossEntropyCriterion):
    """
    Cross entropy loss in combination with Swav loss
    """
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        swav_lambda,
        distributed_world_size,
        rand_factor,
        queue_length,
        swav_epsilon,
        sinkhorn_iterations,
        swav_temperature,
        stability_epsilon=0.0,
        pre_norm_prototypes=True,
        improve_numerical_stability=True,
        update_queue_starts=-1,
        improv_stab_global=False,
        # ----
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        # NOTE: MRO inheritance: SwavMaskedLmLoss -> StdSwavCriteron -> MaskedLmLoss -> FairseqCriterion
        #   call chain: SwavMaskedLmLoss.init -> StdSwavCriteron.init -> MaskedLmLoss.init -> FairseqCriterion.init
        super().__init__(task=task,
            sentence_avg=sentence_avg,
            label_smoothing=label_smoothing, 
            ignore_prefix_size=ignore_prefix_size, 
            report_accuracy=report_accuracy,
            # swav 
            distributed_world_size=distributed_world_size,
            rand_factor=rand_factor,
            queue_length=queue_length,
            swav_epsilon=swav_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            stability_epsilon=stability_epsilon,
            swav_temperature=swav_temperature,
            pre_norm_prototypes=pre_norm_prototypes,
            improve_numerical_stability=improve_numerical_stability,
            update_queue_starts=update_queue_starts,
            improv_stab_global=improv_stab_global,
        )
        self.swav_lambda = swav_lambda
        self.lambda_swav = PiecewiseLinearFn.from_string(swav_lambda)
    
    def infer_lm_swav_lambdas(self, num_updates, **kwargs):
        swp_lambda = self.lambda_swav(num_updates)
        if 0 <= swp_lambda <= 1.0:
            return swp_lambda,  (1.0 - swp_lambda)
        elif swp_lambda < 0:
            raise ValueError
        elif swp_lambda > 1.0:
            assert swp_lambda <= 2.0
            return swp_lambda - 1.0, 1.0
    
    def build_swav_loss(self, **kwargs):
        self._swav = StdSwavCriterion(task=self.task, **kwargs)

    def compute_nll_loss(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        nll_log_output = {}
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            nll_log_output["n_correct"] = utils.item(n_correct.data)
            nll_log_output["total"] = utils.item(total.data)
        return loss, nll_loss, nll_log_output
    
    def compute_sinkhorn_prototypes(self, model, sample, sample_key="net_input", queue=None, no_rand_factor=True, **kwargs):
        # NOTE only use for evaluation, not for use for training or gradient update
        #   otherwise, if decoder exists, training with this fn will cause unused params error   
        extra = model(**sample[sample_key], get_prototypes=True, get_prototypes_only=True)
        rest_extra = {k: v for k, v in extra.items() if k not in ['prot_out', 'prot_embed']}
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        bsz = prot_out.size(0)
        queue = self._swav.get_queue(embed=prot_embed, no_rand_factor=no_rand_factor)
        sinkhorn_prot = self._swav._sinkhorn_prototypes(model.prototypes, prot_out, prot_embed, bsz, queue=queue)
        return sinkhorn_prot, prot_out, prot_embed, rest_extra
    
    def compute_swav_loss_out(self, model, sample):
        # NOTE if prev_output_tokens is not in sample['net_input'], decoder may cause unused params error
        #   if decoder exists, it's not used here, thus raising unused-parameters error, so either:
        #   -> 1. use --find-unused-parameters if use this loss alone (affecting speed)
        #   -> 2. involves the unused-parameters by multiplying with 0 and add to the loss
        #   -> 3. compute cross_entropy loss and add it to swav_loss as final loss
        prototypes_layer = model.prototypes
        extra = model(**sample["net_swav_input"], get_prototypes=True, pre_norm_prototypes=self._swav.pre_norm_prototypes)[1]
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        assert (prot_out is not None and prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {prot_out.size()} / {self._swav.rand_factor}'
        bsz = prot_out.size(0) // self._swav.rand_factor
        queue = self._swav.get_queue(embed=prot_embed, num_updates=getattr(model, 'num_updates', None))
        swav_loss = self._swav.compute_swav_loss(prototypes_layer, prot_out, prot_embed, bsz=bsz, queue=queue)
        return swav_loss

    def forward(self, model, sample, reduce=True, mode="combine"):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert mode == 'combine', f'mode {mode} not supported'
        # assert 0 < self.swav_lambda <= 1.0
        num_updates = getattr(model, 'num_updates', None)
        swp_lambda, nll_lambda = self.infer_lm_swav_lambdas(num_updates=num_updates)

        swav_loss = self.compute_swav_loss_out(model, sample)    
        nll_out_loss, nll_loss, nll_log_output = self.compute_nll_loss(model, sample, reduce=reduce)
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        loss = swav_loss * swp_lambda + nll_lambda * nll_out_loss
        logging_output = {
            "loss": loss.data,
            "swav_loss": swav_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        for k, v in nll_log_output.items():
            logging_output[k] = v
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        swav_loss_sum = sum(log.get("swav_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "swav_loss", swav_loss_sum / len(logging_outputs), len(logging_outputs), round=3
            # "swav_loss", swav_loss_sum / sample_size, sample_size, round=3
        )


@register_criterion(
    "swaved_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig)
class SwavEncDecLabelSmoothedCrossEntropyCriterion(SwavLabelSmoothedCrossEntropyCriterion):
    def compute_swav_loss_out(self, model, sample):
        # NOTE if prev_output_tokens is not in sample['net_input'], decoder may cause unused params error
        #   if decoder exists, it's not used here, thus raising unused-parameters error, so either:
        #   -> 1. use --find-unused-parameters if use this loss alone (affecting speed)
        #   -> 2. involves the unused-parameters by multiplying with 0 and add to the loss
        #   -> 3. compute cross_entropy loss and add it to swav_loss as final loss
        prototypes_layer = model.prototypes
        assert "prev_output_tokens" in sample['net_swav_input']
        extra = model(**sample["net_swav_input"], get_prototypes=True, pre_norm_prototypes=self._swav.pre_norm_prototypes)[1]
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        assert "dec_prot_out" in extra and "dec_prot_embed" in extra
        sec_prot_out = extra['dec_prot_out']
        sec_prot_embed = extra['dec_prot_embed']
        assert (prot_out is not None and prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {prot_out.size()} / {self._swav.rand_factor}'
        assert (sec_prot_out is not None and sec_prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {sec_prot_out.size()} / {self._swav.rand_factor}'
        
        bsz = prot_out.size(0) // self._swav.rand_factor
        queue = self._swav.get_queue(embed=prot_embed, num_updates=getattr(model, 'num_updates', None))
        sec_queue = self._swav.get_queue(embed=sec_prot_embed, num_updates=getattr(model, 'num_updates', None), qname='sec')
        swav_loss = self._swav.compute_swav_loss(
            prototypes_layer, prot_out, prot_embed, bsz=bsz, queue=queue,
            sec_prot_out=sec_prot_out, sec_embedding=sec_prot_embed, sec_queue=sec_queue
        )
        return swav_loss


@register_criterion(
    "langagc1_swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig
)
class LangAgnosticC1SwavLabelSmoothedCrossEntropyCriterion(SwavLabelSmoothedCrossEntropyCriterion):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst1SwavCriterion(task=self.task, **kwargs)
    
    def compute_sinkhorn_prototypes(self, model, sample, sample_key="net_input", queue=None, no_rand_factor=True, **kwargs):
        # NOTE: add `langs` to the equation because lang_agnosticism require langs input
        extra = model(**sample[sample_key], get_prototypes=True, get_prototypes_only=True)
        rest_extra = {k: v for k, v in extra.items() if k not in ['prot_out', 'prot_embed']}
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        bsz = prot_out.size(0)
        langs = sample[sample_key]['src_langs']
        queue = self._swav.get_queue(embed=prot_embed, no_rand_factor=no_rand_factor)
        sinkhorn_prot = self._swav._sinkhorn_prototypes(model.prototypes, prot_out, prot_embed, bsz, langs, queue=queue)
        return sinkhorn_prot, prot_out, prot_embed, rest_extra
    
    def compute_swav_loss_out(self, model, sample):
        # NOTE: add `langs` to the equation because lang_agnosticism require langs input
        prototypes_layer = model.prototypes
        extra = model(**sample["net_swav_input"], get_prototypes=True, pre_norm_prototypes=self._swav.pre_norm_prototypes)[1]
        src_langs = sample["net_swav_input"]['src_langs']
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        assert (prot_out is not None and prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {prot_out.size()} / {self._swav.rand_factor}'
        bsz = prot_out.size(0) // self._swav.rand_factor
        queue = self._swav.get_queue(embed=prot_embed, num_updates=getattr(model, 'num_updates', None))

        num_updates = getattr(model, 'num_updates', None)
        assert num_updates is not None
        swav_loss = self._swav.compute_swav_loss(
            prototypes_layer, prot_out, prot_embed, bsz=bsz, langs=src_langs, queue=queue, 
            # no_constraint=self.constraint_begin_up
        )
        return swav_loss


@register_criterion(
    "langagc1ed_swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig
)
class LangAgnosticC1EncDecSwavLabelSmoothedCrossEntropyCriterion(SwavEncDecLabelSmoothedCrossEntropyCriterion):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst1SwavCriterion(task=self.task, **kwargs)
    
    def compute_swav_loss_out(self, model, sample):
        # NOTE: add `langs` to the equation because lang_agnosticism require langs input
        prototypes_layer = model.prototypes
        extra = model(**sample["net_swav_input"], get_prototypes=True, pre_norm_prototypes=self._swav.pre_norm_prototypes)[1]
        src_langs = sample["net_swav_input"]['src_langs']
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        sec_prot_out = extra['dec_prot_out']
        sec_prot_embed = extra['dec_prot_embed']
        assert (prot_out is not None and prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {prot_out.size()} / {self._swav.rand_factor}'
        assert (sec_prot_out is not None and sec_prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {sec_prot_out.size()} / {self._swav.rand_factor}'
        bsz = prot_out.size(0) // self._swav.rand_factor
        queue = self._swav.get_queue(embed=prot_embed, num_updates=getattr(model, 'num_updates', None))
        sec_queue = self._swav.get_queue(embed=sec_prot_embed, num_updates=getattr(model, 'num_updates', None), qname='sec')
        swav_loss = self._swav.compute_swav_loss(prototypes_layer, prot_out, prot_embed, bsz=bsz, langs=src_langs, queue=queue,
            sec_prot_out=sec_prot_out, sec_embedding=sec_prot_embed, sec_queue=sec_queue
        )
        return swav_loss


@register_criterion(
    "langagc1preexp_swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig
)
class LangAgnosticC1PreExpSwavLabelSmoothedCrossEntropyCriterion(LangAgnosticC1SwavLabelSmoothedCrossEntropyCriterion):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst1PreExpSwavCriterion(task=self.task, **kwargs)


@register_criterion(
    "langagc2_swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig
)
class LangAgnosticC2SwavLabelSmoothedCrossEntropyCriterion(LangAgnosticC1SwavLabelSmoothedCrossEntropyCriterion):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst2SwavCriterion(task=self.task, **kwargs)


@register_criterion(
    "langagc3_swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig
)
class LangAgnosticC3SwavLabelSmoothedCrossEntropyCriterion(LangAgnosticC1SwavLabelSmoothedCrossEntropyCriterion):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst3SwavCriterion(task=self.task, **kwargs)


@register_criterion(
    "langagc3preexp_swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig
)
class LangAgnosticC3PreExpSwavLabelSmoothedCrossEntropyCriterion(LangAgnosticC1SwavLabelSmoothedCrossEntropyCriterion):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst3PreExpSwavCriterion(task=self.task, **kwargs)


# NOTE: Final Language Agnostic constraint for swav loss
@register_criterion(
    "langagc3preexpplus_swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig
)
class LangAgnosticC3PreExpPlusSwavLabelSmoothedCrossEntropyCriterion(LangAgnosticC1SwavLabelSmoothedCrossEntropyCriterion):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst3PreExpPlusSwavCriterion(task=self.task, **kwargs)


# NOTE: Final Language Agnostic constraint for swav loss
@register_criterion(
    "freq_langag_swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig
)
class FreqLangAgnosticConstSwavLabelSmoothedCrossEntropyCriterion(LangAgnosticC3PreExpPlusSwavLabelSmoothedCrossEntropyCriterion):
    def build_swav_loss(self, **kwargs):
        self._swav = FreqLangAgnosticConstSwavCriterion(task=self.task, **kwargs)


@register_criterion(
    "langagc3preexpplus_ed_swav_label_smoothed_cross_entropy", dataclass=SwavLabelSmoothedCrossEntropyCriterionConfig
)
class LangAgnosticC3PreExpPlusEncDecSwavLabelSmoothedCrossEntropyCriterion(LangAgnosticC1EncDecSwavLabelSmoothedCrossEntropyCriterion):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst3PreExpPlusSwavCriterion(task=self.task, **kwargs)


@dataclass
class SwavMaskedLmCriterionConfig(MaskedLmConfig, StdSwavConfig):
    swav_lambda: str = field(
        default="0.5",
        metadata={"help": "swav lambda, if < 1: loss=(1.0 - lambda)lm + lambda*swav, if > 1: loss=lm + (lambda - 1)*swav"},
    )


@register_criterion(
    "swav_masked_lm", dataclass=SwavMaskedLmCriterionConfig
)
class SwavMaskedLmLoss(SwavCriterionWrapper, MaskedLmLoss):
    """Standard Masked LM loss in combination with swav loss
    There different mode:
        "combine": combine MLM and swav losses into 1 with lambda_swav
        "swav": only involve SWAV loss, beware of unused MLM params
        "mlm": only involve MLM loss, beware of unused SWAV params
    """
    def __init__(
        self,
        cfg,
        task,
        swav_lambda,
        distributed_world_size,
        rand_factor,
        queue_length,
        swav_epsilon,
        sinkhorn_iterations,
        swav_temperature,
        stability_epsilon=0.0,
        pre_norm_prototypes=True,
        improve_numerical_stability=True,
        update_queue_starts=-1,
        improv_stab_global=False,
    ):
        # NOTE: MRO inheritance: SwavMaskedLmLoss -> SwavCriterionWrapper -> MaskedLmLoss -> FairseqCriterion
        super().__init__(
            cfg=cfg,
            task=task,
            distributed_world_size=distributed_world_size,
            rand_factor=rand_factor,
            queue_length=queue_length,
            swav_epsilon=swav_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            stability_epsilon=stability_epsilon,
            swav_temperature=swav_temperature,
            pre_norm_prototypes=pre_norm_prototypes,
            improve_numerical_stability=improve_numerical_stability,
            update_queue_starts=update_queue_starts,
            improv_stab_global=improv_stab_global,
        )
        self.swav_lambda = swav_lambda
        self.lambda_swav = PiecewiseLinearFn.from_string(swav_lambda)
    
    def infer_lm_swav_lambdas(self, num_updates, **kwargs):
        swp_lambda = self.lambda_swav(num_updates)
        if 0 <= swp_lambda <= 1.0:
            return swp_lambda,  (1.0 - swp_lambda)
        elif swp_lambda < 0:
            raise ValueError(f'{self.swav_lambda} invalid {swp_lambda}')
        elif swp_lambda > 1.0:
            assert swp_lambda <= 2.0
            return swp_lambda - 1.0, 1.0
    
    def build_swav_loss(self, **kwargs):
        self._swav = StdSwavCriterion(task=self.task, **kwargs)
    
    def compute_sinkhorn_prototypes(self, model, sample, sample_key="net_input", queue=None, no_rand_factor=True, **kwargs):
        extra = model(**sample[sample_key], get_prototypes=True, get_prototypes_only=True)
        rest_extra = {k: v for k, v in extra.items() if k != 'prot_out' and k != 'prot_embed'}
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        bsz = prot_out.size(0)
        queue = self._swav.get_queue(embed=prot_embed, no_rand_factor=no_rand_factor)
        sinkhorn_prot = self._swav._sinkhorn_prototypes(model.prototypes, prot_out, prot_embed, bsz, queue=queue)
        return sinkhorn_prot, prot_out, prot_embed, rest_extra
    
    def compute_swav_loss_out(self, model, sample):
        # NOTE : potential unused-parameters error issue, please do either:
        #   1. compute swav loss along with xentropy_loss and add them together to prevent unused-params
        #   2. multiplying output_layer (projection) with 0 and add to the loss if using stand-along swav_loss
        prototypes_layer = model.prototypes
        extra = model(**sample["net_swav_input"], get_prototypes=True, pre_norm_prototypes=self._swav.pre_norm_prototypes)[1]
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        assert (prot_out is not None and prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {prot_out.size()} / {self._swav.rand_factor}'
        bsz = prot_out.size(0) // self._swav.rand_factor
        queue = self._swav.get_queue(embed=prot_embed, num_updates=getattr(model, 'num_updates', None))
        swav_loss = self._swav.compute_swav_loss(prototypes_layer, prot_out, prot_embed, bsz=bsz, queue=queue)
        return swav_loss
    
    def compute_xentropy_loss_out(self, model, sample):
        # NOTE : potential unused-parameters error issue, please do either:
        #   1. compute swav loss along with xentropy_loss and add them together to prevent unused-params
        #   2. multiplying `prototypes` parameters with 0 and add to the loss if using stand-along xentropy
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )
        
        logits = model(**sample["net_input"], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        masked_lm_loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )
        return masked_lm_loss, sample_size
    
    def forward(self, model, sample, reduce=True, mode="combine"):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if mode == "combine":
            num_updates = getattr(model, 'num_updates', None)
            swp_lambda, nll_lambda = self.infer_lm_swav_lambdas(num_updates=num_updates)
            device = sample['net_input']['src_tokens'].device
            swav_loss = self.compute_swav_loss_out(model, sample) if swp_lambda > 0.0 else torch.FloatTensor([0.0]).to(device=device).sum()
            mlm_loss, mlm_sample_size = self.compute_xentropy_loss_out(model, sample) if nll_lambda > 0.0 else (torch.FloatTensor([0.0]).to(device=device).sum(), 1)
            sample_size = mlm_sample_size
            loss = swav_loss * swp_lambda + nll_lambda * mlm_loss
            logging_output = {
                "loss": loss if self.tpu else loss.data,
                "swav_loss": swav_loss if self.tpu else swav_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["nsentences"],
                "sample_size": sample_size,
            }
        elif mode == "swav":
            loss = self.compute_swav_loss_out(model, sample)
            sample_size = sample["net_swav_input"]["src_tokens"].size(0)
            logging_output = {
                "loss": loss if self.tpu else loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["nsentences"],
                "sample_size": sample_size,
            }
        elif mode == "lm" or mode == "mlm":
            loss, sample_size = self.compute_xentropy_loss_out(model, sample)
            logging_output = {
                "loss": loss if self.tpu else loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["nsentences"],
                "sample_size": sample_size,
            }
        else:
            raise ValueError(f'mode {mode} not found.')
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        MaskedLmLoss.reduce_metrics(logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        swav_loss_sum = sum(log.get("swav_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "swav_loss", swav_loss_sum / len(logging_outputs), len(logging_outputs), round=3
        )


@register_criterion(
    "langagc1_swav_masked_lm", dataclass=SwavMaskedLmCriterionConfig
)
class LangAgnosticC1SwavMaskedLmLoss(SwavMaskedLmLoss):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst1SwavCriterion(task=self.task, **kwargs)
    
    def compute_sinkhorn_prototypes(self, model, sample, sample_key="net_input", queue=None, no_rand_factor=True, **kwargs):
        # NOTE: add langs 
        extra = model(**sample[sample_key], get_prototypes=True, get_prototypes_only=True)
        rest_extra = {k: v for k, v in extra.items() if k != 'prot_out' and k != 'prot_embed'}
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        bsz = prot_out.size(0)
        langs = sample[sample_key]['src_langs']
        queue = self._swav.get_queue(embed=prot_embed, no_rand_factor=no_rand_factor)
        sinkhorn_prot = self._swav._sinkhorn_prototypes(model.prototypes, prot_out, prot_embed, bsz, langs, queue=queue)
        return sinkhorn_prot, prot_out, prot_embed, rest_extra

    def compute_swav_loss_out(self, model, sample):
        prototypes_layer = model.prototypes
        extra = model(**sample["net_swav_input"], get_prototypes=True, pre_norm_prototypes=self._swav.pre_norm_prototypes)[1]
        src_langs = sample["net_swav_input"]['src_langs']
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        assert (prot_out is not None and prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {prot_out.size()} / {self._swav.rand_factor}'
        bsz = prot_out.size(0) // self._swav.rand_factor
        queue = self._swav.get_queue(embed=prot_embed, num_updates=getattr(model, 'num_updates', None))
        swav_loss = self._swav.compute_swav_loss(prototypes_layer, prot_out, prot_embed, bsz=bsz, langs=src_langs, queue=queue)
        return swav_loss


@register_criterion(
    "langagc2_swav_masked_lm", dataclass=SwavMaskedLmCriterionConfig
)
class LangAgnosticC2SwavMaskedLmLoss(LangAgnosticC1SwavMaskedLmLoss):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst2SwavCriterion(task=self.task, **kwargs)


@register_criterion(
    "langagc3_swav_masked_lm", dataclass=SwavMaskedLmCriterionConfig
)
class LangAgnosticC3SwavMaskedLmLoss(LangAgnosticC1SwavMaskedLmLoss):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst3SwavCriterion(task=self.task, **kwargs)


@register_criterion(
    "langagc3preexp_swav_masked_lm", dataclass=SwavMaskedLmCriterionConfig
)
class LangAgnosticC3PreExpSwavMaskedLmLoss(LangAgnosticC1SwavMaskedLmLoss):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst3PreExpSwavCriterion(task=self.task, **kwargs)


@register_criterion(
    "langagc3preexpplus_swav_masked_lm", dataclass=SwavMaskedLmCriterionConfig
)
class LangAgnosticC3PreExpPlusSwavMaskedLmLoss(LangAgnosticC1SwavMaskedLmLoss):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst3PreExpPlusSwavCriterion(task=self.task, **kwargs)


# NOTE: FINAL language agnostic constraint
@register_criterion(
    "freq_langag_swav_masked_lm", dataclass=SwavMaskedLmCriterionConfig
)
class FreqLangAgnosticConstSwavMaskedLmLoss(LangAgnosticC1SwavMaskedLmLoss):
    def build_swav_loss(self, **kwargs):
        self._swav = FreqLangAgnosticConstSwavCriterion(task=self.task, **kwargs)


@register_criterion(
    "weighted_label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class WeightedLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        weights = sample.get('weights', None)
        if weights is not None:
            assert weights.size(0) == sample["target"].size(0), f'debug strict: {weights.size()=}/{sample["target"].size(0)=} {weights}'
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, weights=weights)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
    
    def get_lprobs_and_target(self, model, net_output, sample, weights=None):
        # lprobs: [bsz, slen, d]
        # target: [bsz, slen, d]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        assert lprobs.dim() == 3, f'{lprobs.size()}'
        if weights is not None:
            lprobs = lprobs * weights[:, None, None].type_as(lprobs).to(lprobs.device)
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, weights=None):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, weights=weights)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss



@dataclass
class SwavWav2VecCriterionConfig(Wav2VecCriterionConfig, StdSwavConfig):
    swav_lambda: str = field(
        default="0.5",
        metadata={"help": "swav lambda, if < 1: loss=(1.0 - lambda)lm + lambda*swav, if > 1: loss=lm + (lambda - 1)*swav"},
    )


@register_criterion(
    "swav_wav2vec", dataclass=SwavWav2VecCriterionConfig
)
class SwavWav2VecLoss(SwavCriterionWrapper, Wav2vecCriterion):
    def __init__(
        self,
        task,
        swav_lambda,
        distributed_world_size,
        rand_factor,
        queue_length,
        swav_epsilon,
        sinkhorn_iterations,
        swav_temperature,
        stability_epsilon=0.0,
        pre_norm_prototypes=True,
        improve_numerical_stability=True,
        update_queue_starts=-1,
        improv_stab_global=False,
        # ---
        infonce=False, loss_weights=None, log_keys=None
    ):
        # NOTE: MRO inheritance: SwavMaskedLmLoss -> SwavCriterionWrapper -> MaskedLmLoss -> FairseqCriterion
        super().__init__(
            task=task,
            infonce=infonce,
            loss_weights=loss_weights,
            log_keys=log_keys,
            # 
            distributed_world_size=distributed_world_size,
            rand_factor=rand_factor,
            queue_length=queue_length,
            swav_epsilon=swav_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            stability_epsilon=stability_epsilon,
            swav_temperature=swav_temperature,
            pre_norm_prototypes=pre_norm_prototypes,
            improve_numerical_stability=improve_numerical_stability,
            update_queue_starts=update_queue_starts,
            improv_stab_global=improv_stab_global,
        )
        self.swav_lambda = swav_lambda
        self.lambda_swav = PiecewiseLinearFn.from_string(swav_lambda)
    
    def infer_lm_swav_lambdas(self, num_updates, **kwargs):
        # NOTE: lambda should be > 1
        swp_lambda = self.lambda_swav(num_updates)
        if 0 <= swp_lambda <= 1.0:
            return swp_lambda,  (1.0 - swp_lambda)
        elif swp_lambda < 0:
            raise ValueError(f'{self.swav_lambda} invalid {swp_lambda}')
        elif swp_lambda > 1.0:
            assert swp_lambda <= 2.0
            return swp_lambda - 1.0, 1.0
    
    def build_swav_loss(self, **kwargs):
        self._swav = StdSwavCriterion(task=self.task, **kwargs)
    
    def compute_sinkhorn_prototypes(self, model, sample, sample_key="net_input", queue=None, no_rand_factor=True, **kwargs):
        extra = model(**sample[sample_key], get_prototypes=True, get_prototypes_only=True)
        rest_extra = {k: v for k, v in extra.items() if k != 'prot_out' and k != 'prot_embed'}
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        bsz = prot_out.size(0)
        queue = self._swav.get_queue(embed=prot_embed, no_rand_factor=no_rand_factor)
        sinkhorn_prot = self._swav._sinkhorn_prototypes(model.prototypes, prot_out, prot_embed, bsz, queue=queue)
        return sinkhorn_prot, prot_out, prot_embed, rest_extra
    
    def compute_swav_loss_out(self, model, sample):
        # NOTE : potential unused-parameters error issue, please do either:
        #   1. compute swav loss along with xentropy_loss and add them together to prevent unused-params
        #   2. multiplying output_layer (projection) with 0 and add to the loss if using stand-along swav_loss
        prototypes_layer = model.prototypes
        extra = model(**sample["net_swav_input"], get_prototypes=True, pre_norm_prototypes=self._swav.pre_norm_prototypes)
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        assert (prot_out is not None and prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {prot_out.size()} / {self._swav.rand_factor}'
        bsz = prot_out.size(0) // self._swav.rand_factor
        queue = self._swav.get_queue(embed=prot_embed, num_updates=getattr(model, 'num_updates', None))
        swav_loss = self._swav.compute_swav_loss(prototypes_layer, prot_out, prot_embed, bsz=bsz, queue=queue)
        return swav_loss
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)
        self.xla = is_xla_tensor(logits)

        # XXX: handle weights on xla.
        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        reduction = "none" if ((not reduce) or self.xla) else "sum"
        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction=reduction)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), weights, reduction=reduction
            )

        if self.xla:
            # tpu-comment: since dynamic shapes lead to recompilations on xla,
            # we don't shrink tensors using mask_indices.
            # Instead, we use mask indices to adjust loss.
            mi = (
                sample['net_input']['mask_indices']
                .transpose(0, 1)  # logits are transposed in `model.get_logits`
                .reshape(logits.size(0))
            )
            loss = (loss * mi).sum() if reduce else (loss * mi)

        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        # swav loss
        num_updates = getattr(model, 'num_updates', None)
        swp_lambda, nll_lambda = self.infer_lm_swav_lambdas(num_updates=num_updates)
        device = loss.device
        swav_loss = self.compute_swav_loss_out(model, sample) if swp_lambda > 0.0 else torch.FloatTensor([0.0]).to(device=device).sum()

        loss = loss + swav_loss

        logging_output = {
            "loss": loss.item() if (reduce and not self.xla) else loss.detach(),
            "swav_loss": swav_loss.item() if (reduce and not self.xla) else swav_loss.detach(),
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(model, "get_original_targets"):
                        original_target = model.get_original_targets(sample, net_output)
                    else:
                        original_target = target
                    logging_output["target"] = original_target.cpu().numpy()
            elif lk in net_output:
                value = net_output[lk]
                if not is_xla_tensor(value):
                    value = float(value)
                logging_output[lk] = value

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item() if not self.xla else l.detach()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    if is_xla_tensor(logits):
                        max, min = max * mi, min * mi
                        both = max & min
                        corr = max.long().sum() - both.long().sum()
                        count = mi.sum()
                    else:
                        both = max & min
                        corr = max.long().sum().item() - both.long().sum().item()
                        count = float(max.numel())

                logging_output["correct"] = corr
                logging_output["count"] = count

        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "correct",
            "count",
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss"):
                    metrics.log_scalar(
                        k, val / (sample_size or 1) / math.log(2), sample_size, round=3
                    )
                else:
                    metrics.log_scalar(k, val / len(logging_outputs), round=3)

        swav_loss_sum = sum(log.get("swav_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "swav_loss", swav_loss_sum / len(logging_outputs), len(logging_outputs), round=3
        )




@register_criterion(
    "langagc1_swav_wav2vec", dataclass=SwavWav2VecCriterionConfig
)
class LangAgnosticC1SwavWav2VecLoss(SwavWav2VecLoss):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst1SwavCriterion(task=self.task, **kwargs)
    
    def compute_sinkhorn_prototypes(self, model, sample, sample_key="net_input", queue=None, no_rand_factor=True, **kwargs):
        # NOTE: add langs 
        extra = model(**sample[sample_key], get_prototypes=True, get_prototypes_only=True)
        rest_extra = {k: v for k, v in extra.items() if k != 'prot_out' and k != 'prot_embed'}
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        bsz = prot_out.size(0)
        langs = sample[sample_key]['src_langs']
        queue = self._swav.get_queue(embed=prot_embed, no_rand_factor=no_rand_factor)
        sinkhorn_prot = self._swav._sinkhorn_prototypes(model.prototypes, prot_out, prot_embed, bsz, langs, queue=queue)
        return sinkhorn_prot, prot_out, prot_embed, rest_extra

    def compute_swav_loss_out(self, model, sample):
        prototypes_layer = model.prototypes
        extra = model(**sample["net_swav_input"], get_prototypes=True, pre_norm_prototypes=self._swav.pre_norm_prototypes)
        src_langs = sample["net_swav_input"]['src_langs']
        prot_out = extra['prot_out']
        prot_embed = extra['prot_embed']
        assert (prot_out is not None and prot_out.size(0) % self._swav.rand_factor == 0
            ), f'rand_fac wrong {prot_out.size()} / {self._swav.rand_factor}'
        bsz = prot_out.size(0) // self._swav.rand_factor
        queue = self._swav.get_queue(embed=prot_embed, num_updates=getattr(model, 'num_updates', None))
        swav_loss = self._swav.compute_swav_loss(prototypes_layer, prot_out, prot_embed, bsz=bsz, langs=src_langs, queue=queue)
        return swav_loss




@register_criterion(
    "langagc3_swav_wav2vec", dataclass=SwavWav2VecCriterionConfig
)
class LangAgnosticC3SwavWav2VecLoss(LangAgnosticC1SwavWav2VecLoss):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst3SwavCriterion(task=self.task, **kwargs)


@register_criterion(
    "langagc3preexpplus_swav_wav2vec", dataclass=SwavWav2VecCriterionConfig
)
class LangAgnosticC3PreExpPlusSwavWav2VecLoss(LangAgnosticC1SwavWav2VecLoss):
    def build_swav_loss(self, **kwargs):
        self._swav = LangAgnosticConst3PreExpPlusSwavCriterion(task=self.task, **kwargs)


