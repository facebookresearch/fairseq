# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
from itertools import chain
import logging
import sys
from typing import Any, Dict, List

import torch

from fairseq import checkpoint_utils, distributed_utils, models, optim, utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.nan_detector import NanDetector
from fairseq.optim import lr_scheduler


logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, quantizer=None):
        self.args = args
        self.task = task

        # catalog shared parameters
        shared_params = _catalog_shared_params(model)

        self.tpu = getattr(args, 'tpu', False)
        self.cuda = torch.cuda.is_available() and not args.cpu and not self.tpu
        if self.cuda:
            self.device = torch.device('cuda')
        elif self.tpu:
            self.device = utils.get_tpu_device(args)
        else:
            self.device = torch.device('cpu')

        # copy model and criterion to current device/dtype
        self._criterion = criterion
        self._model = model
        if self.tpu:
            import torch_xla.core.xla_model as xm
            self._model = xm.send_cpu_data_to_device(self._model, self.device)
        if args.fp16:
            self._criterion = self._criterion.half()
            self._model = self._model.half()
        elif args.bf16:
            self._criterion = self._criterion.to(dtype=torch.bfloat16)
            self._model = self._model.to(dtype=torch.bfloat16)
        self._criterion = self._criterion.to(device=self.device)
        self._model = self._model.to(device=self.device)

        # check that shared parameters are preserved after device transfer
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info(
                    'detected shared parameter: {} <- {}'.format(shared_param[0], path)
                )
                _set_module_by_path(self._model, path, ref)

        self._dummy_batch = "DUMMY"  # indicates we don't have a dummy batch at first
        self._lr_scheduler = None
        self._num_updates = 0
        self._num_xla_compiles = 0  # for TPUs
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None

        # TODO(myleott): support tpu
        if self.cuda and self.data_parallel_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(self.data_parallel_world_size)
        else:
            self._grad_norm_buf = None

        self.quantizer = quantizer
        if self.quantizer is not None:
            self.quantizer.set_trainer(self)

        # get detailed cuda environment
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(self.cuda_env)
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=0)

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change."""
        self._lr_scheduler = None
        self._optimizer = None
        self._wrapped_criterion = None
        self._wrapped_model = None

    @property
    def data_parallel_world_size(self):
        return self.args.distributed_world_size

    @property
    def data_parallel_process_group(self):
        if self.tpu:
            return ('tpu', None)
        else:
            return None

    @property
    def data_parallel_rank(self):
        return self.args.distributed_rank

    @property
    def is_data_parallel_master(self):
        return distributed_utils.is_master(self.args)

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if (
                utils.has_parameters(self._criterion)
                and self.data_parallel_world_size > 1
                and not self.args.use_bmuf
                and not self.tpu
            ):
                self._wrapped_criterion = models.DistributedFairseqModel(
                    self.args, self._criterion,
                    process_group=self.data_parallel_process_group
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if (
                self.data_parallel_world_size > 1
                and not self.args.use_bmuf
                and not self.tpu
            ):
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                    process_group=self.data_parallel_process_group
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )

        if self.args.fp16 or self.args.bf16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16, "
                    "please switch to FP32 which is likely to be faster"
                )
            if self.args.memory_efficient_fp16 or self.args.memory_efficient_bf16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                    self.args, params
                )
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info("NOTE: your device may support faster training with --fp16")
            self._optimizer = optim.build_optimizer(self.args, params)

        if self.args.use_bmuf:
            self._optimizer = optim.FairseqBMUF(self.args, self._optimizer)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if self.is_data_parallel_master:  # only save one checkpoint
            extra_state["metrics"] = metrics.state_dict()
            checkpoint_utils.save_state(
                filename,
                self.args,
                self.get_model().state_dict(),
                self.get_criterion(),
                self.optimizer,
                self.lr_scheduler,
                self.get_num_updates(),
                self._optim_history,
                extra_state,
            )

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = None, [], None

        bexists = PathManager.isfile(filename)
        if bexists:
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)

            # load model parameters
            try:
                self.get_model().load_state_dict(
                    state["model"], strict=True, args=self.args
                )
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )

            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]
            last_optim_state = state.get("last_optimizer_state", None)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), "Criterion does not match; please reset the optimizer (--reset-optimizer)."
            assert (
                last_optim["optimizer_name"] == self.optimizer.__class__.__name__
            ), "Optimizer does not match; please reset the optimizer (--reset-optimizer)."

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            epoch = extra_state["train_iterator"]["epoch"]
            logger.info(
                "loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

            self.lr_step(epoch)

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()
        else:
            logger.info("no existing checkpoint found {}".format(filename))

        return extra_state

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            logger.info("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.args.train_subset,
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
            )
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.args.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=self.data_parallel_rank if shard_batch_itr else 0,
            num_workers=self.args.num_workers,
            epoch=epoch
        )

    def get_valid_iterator(
        self,
        subset,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.args.max_tokens_valid,
            max_sentences=self.args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.args.num_workers
        )

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        if self.quantizer is not None:
            self.quantizer.begin_epoch(epoch)

        # task specific setup per epoch
        self.task.begin_epoch(epoch, self.get_model())

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""
        if self._dummy_batch == "DUMMY":
            self._dummy_batch = samples[0]

        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=0)

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                is_dummy_batch = True
            else:
                is_dummy_batch = False

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.data_parallel_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample=sample,
                        model=self.model,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        update_num=self.get_num_updates(),
                        ignore_grad=is_dummy_batch,
                    )
                    del loss

                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    logger.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    ooms += 1
                    self.zero_grad()
                    if self.cuda:
                        torch.cuda.empty_cache()
                    if self.args.distributed_world_size == 1:
                        return None
                else:
                    raise e

            if self.tpu and i < len(samples) - 1:
                # tpu-comment: every XLA operation before marking step is
                # appended to the IR graph, and processing too many batches
                # before marking step can lead to OOM errors.
                # To handle gradient accumulation use case, we explicitly
                # mark step here for every forward pass without a backward pass
                import torch_xla.core.xla_model as xm
                xm.mark_step()

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # gather logging outputs from all replicas
        if self._sync_stats():
            logging_outputs, (sample_size, ooms) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ooms, ignore=is_dummy_batch,
            )

        overflow = False
        try:
            if self.tpu and self.data_parallel_world_size > 1:
                import torch_xla.core.xla_model as xm
                gradients = xm._fetch_gradients(self.optimizer.optimizer)
                xm.all_reduce('sum', gradients, scale=1.0 / self.data_parallel_world_size)

            with torch.autograd.profiler.record_function("multiply-grads"):
                # multiply gradients by (# GPUs / sample_size) since DDP
                # already normalizes by the number of GPUs. Thus we get
                # (sum_of_gradients / sample_size).
                if not self.args.use_bmuf:
                    self.optimizer.multiply_grads(self.data_parallel_world_size / sample_size)
                elif sample_size > 0:  # BMUF needs to check sample size
                    num = self.data_parallel_world_size if self._sync_stats() else 1
                    self.optimizer.multiply_grads(num / sample_size)

            with torch.autograd.profiler.record_function("clip-grads"):
                # clip grads
                grad_norm = self.clip_grad_norm(self.args.clip_norm)

            # check that grad norms are consistent across workers
            if (
                not self.args.use_bmuf
                and self.args.distributed_wrapper != 'SlowMo'
                and not self.tpu
            ):
                self._check_grad_norms(grad_norm)

            with torch.autograd.profiler.record_function("optimizer"):
                # take an optimization step
                self.optimizer.step()
        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print
            # out where it fails
            with NanDetector(self.model):
                self.task.train_step(
                    sample, self.model, self.criterion, self.optimizer, self.get_num_updates(),
                    ignore_grad=False
                )
            raise
        except OverflowError as e:
            overflow = True
            logger.info("NOTE: overflow detected, " + str(e))
            grad_norm = torch.tensor(0.).cuda()
            self.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise e

        # Some distributed wrappers (e.g., SlowMo) need access to the optimizer after the step
        if hasattr(self.model, 'perform_additional_optimizer_actions'):
            if hasattr(self.optimizer, 'fp32_params'):
                self.model.perform_additional_optimizer_actions(self.optimizer.optimizer, self.optimizer.fp32_params)
            else:
                self.model.perform_additional_optimizer_actions(self.optimizer.optimizer)

        if not overflow or self.args.distributed_wrapper == 'SlowMo':
            self.set_num_updates(self.get_num_updates() + 1)

            if self.tpu:
                # mark step on TPUs
                import torch_xla.core.xla_model as xm
                xm.mark_step()

                # only log stats every log_interval steps
                # this causes wps to be misreported when log_interval > 1
                logging_output = {}
                if self.get_num_updates() % self.args.log_interval == 0:
                    logging_output = self._reduce_and_log_stats(
                        logging_outputs, sample_size, grad_norm,
                    )

                # log whenever there's an XLA compilation, since these
                # slow down training and may indicate opportunities for
                # optimization
                self._check_xla_compilation()
            else:
                # log stats
                logging_output = self._reduce_and_log_stats(
                    logging_outputs, sample_size, grad_norm,
                )

                # clear CUDA cache to reduce memory fragmentation
                if (
                    self.cuda
                    and self.args.empty_cache_freq > 0
                    and (
                        (self.get_num_updates() + self.args.empty_cache_freq - 1)
                        % self.args.empty_cache_freq
                    ) == 0
                ):
                    torch.cuda.empty_cache()

        if self.args.fp16:
            metrics.log_scalar("loss_scale", self.optimizer.scaler.loss_scale, priority=700, round=0)

        metrics.log_stop_time("train_wall")

        return logging_output

    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample
        if self.tpu:
            import torch_xla.core.xla_model as xm
            xm.rendezvous('valid_step')  # wait for all workers
            xm.mark_step()

        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                is_dummy_batch = True
            else:
                is_dummy_batch = False

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        logger.warning(
                            "ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            logging_outputs = [logging_output]
            if is_dummy_batch:
                if torch.is_tensor(sample_size):
                    sample_size.zero_()
                else:
                    sample_size *= 0.

        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            logging_outputs, (sample_size, ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ignore=is_dummy_batch,
            )

        # log validation stats
        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)

        return logging_output

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_meter(self, name):
        """[deprecated] Get a specific meter by name."""
        from fairseq import meters

        if 'get_meter' not in self._warn_once:
            self._warn_once.add('get_meter')
            utils.deprecation_warning(
                'Trainer.get_meter is deprecated. Please use fairseq.metrics instead.'
            )

        train_meters = metrics.get_meters("train")
        if train_meters is None:
            train_meters = {}

        if name == "train_loss" and "loss" in train_meters:
            return train_meters["loss"]
        elif name == "train_nll_loss":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = train_meters.get("nll_loss", None)
            return m or meters.AverageMeter()
        elif name == "wall":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = metrics.get_meter("default", "wall")
            return m or meters.TimeMeter()
        elif name == "wps":
            m = metrics.get_meter("train", "wps")
            return m or meters.TimeMeter()
        elif name in {"valid_loss", "valid_nll_loss"}:
            # support for legacy train.py, which assumed these meters
            # are always initialized
            k = name[len("valid_"):]
            m = metrics.get_meter("valid", k)
            return m or meters.AverageMeter()
        elif name == "oom":
            return meters.AverageMeter()
        elif name in train_meters:
            return train_meters[name]
        return None

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        if self.quantizer:
            self.quantizer.step_update(self._num_updates)
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def clip_grad_norm(self, clip_norm):
        return self.optimizer.clip_grad_norm(clip_norm, aggregate_norm_fn=None)

    def _prepare_sample(self, sample):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            return None

        if self.cuda:
            sample = utils.move_to_cuda(sample)

        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        if self.args.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        if self.args.bf16:
            sample = utils.apply_to_sample(apply_bfloat16, sample)

        return sample

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP or multiple GPUs with
        # BMUF and it's a bmuf sync with warmup iterations completed before.
        if self.data_parallel_world_size == 1:
            return False
        elif self.args.use_bmuf:
            return (
                (self.get_num_updates() + 1) % self.args.global_sync_iter == 0
                and (self.get_num_updates() + 1) > self.args.warmup_iterations
            )
        else:
            return True

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        if self.task.__class__.logging_outputs_can_be_summed(self.get_criterion()):
            return self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            return self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if self.tpu:
            raise NotImplementedError
        if ignore:
            logging_outputs = []
        results = list(zip(
            *distributed_utils.all_gather_list(
                [logging_outputs] + list(extra_stats_to_sum),
                max_size=getattr(self.args, 'all_gather_list_size', 16384),
                group=self.data_parallel_process_group,
            )
        ))
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data['extra_stats_' + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data['logging_outputs_' + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data,
            device=self.device,
            group=self.data_parallel_process_group
        )

        extra_stats_to_sum = [
            data['extra_stats_' + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data['logging_outputs_' + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.data_parallel_rank] = grad_norm
            distributed_utils.all_reduce(
                self._grad_norm_buf,
                group=self.data_parallel_process_group
            )

            def is_consistent(tensor):
                max_abs_diff = torch.max(torch.abs(tensor - tensor[0]))
                return (max_abs_diff / (tensor[0] + 1e-6) < 1e-6).all()

            if not is_consistent(self._grad_norm_buf):
                pretty_detail = "\n".join(
                    "rank {:3d} = {:.8f}".format(r, n)
                    for r, n in enumerate(self._grad_norm_buf.tolist())
                )
                error_detail = "grad_norm across the workers:\n{}\n".format(pretty_detail)
                raise RuntimeError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=no_c10d. "
                    "Or are you mixing up different generation of GPUs in training?"
                    + "\n"
                    + "-" * 80
                    + "\n{}\n".format(error_detail)
                    + "-" * 80
                )

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
        if grad_norm is not None:
            metrics.log_speed("ups", 1., priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.args.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.args.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                )

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.get_criterion())
                del logging_outputs

            # extra warning for criterions that don't properly log a loss value
            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Criterion.reduce_metrics did not log a 'loss' value, "
                        "which may break some functionality"
                    )
                metrics.log_scalar("loss", -1)

            # support legacy interface
            if self.tpu:
                logging_output = {}
            else:
                logging_output = agg.get_smoothed_values()
                logging_output["sample_size"] = sample_size
                for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                    if key_to_delete in logging_output:
                        del logging_output[key_to_delete]
            return logging_output

    def _check_xla_compilation(self, message=None):
        import torch_xla.debug.metrics as met
        compile_stats = met.metric_data("CompileTime")
        if compile_stats is None:
            return
        num_xla_compiles = compile_stats[0]
        if num_xla_compiles > self._num_xla_compiles:
            if message is None:
                message = (
                    "too many of these can lead to slow training, "
                    "but we expect a few in the beginning"
                )
            logging.info("NOTE: XLA compilation detected; {}".format(message))
        self._num_xla_compiles = num_xla_compiles


def _catalog_shared_params(module, memo=None, prefix=''):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ('.' if prefix else '') + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def _get_module_by_path(module, path):
    path = path.split('.')
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path = path.split('.')
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)
