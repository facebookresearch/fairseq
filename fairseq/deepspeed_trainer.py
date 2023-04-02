"""
"""
import os
import json
import time
import torch
import logging
import subprocess
import deepspeed
from itertools import chain

from fairseq import checkpoint_utils, models, optim, utils
from fairseq.trainer import Trainer
from fairseq.optim import lr_scheduler
from fairseq.dataclass.configs import FairseqConfig
from fairseq.logging import meters, metrics
from fairseq.optim.dynamic_loss_scaler import DynamicLossScaler

logger = logging.getLogger(__name__)

def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }

    return split_params_into_different_moe_groups_for_optimizer(parameters)

class DeepSpeedTrainer(Trainer):
    def __init__(self, cfg: FairseqConfig, task, model, criterion, quantizer=None):
        assert quantizer is None, "fairseq quantizer is not yet supported by deepspeed"

        super().__init__(cfg, task, model, criterion)

        self.train_step_count = 0

        #TODO: figure out workaround for checkpoint copying to not require rsync
        #try:
        #     subprocess.check_output('which rsync', shell=True)
        #except subprocess.CalledProcessError:
        #    raise RuntimeError('Please install rsync, this is required for model checkpointing')

        ds_config = {}
        if cfg.common.ds_config:
            assert os.path.isfile(cfg.common.ds_config), f"deepspeed config path is not a file: {cfg.common.ds_config}"
            with open(cfg.common.ds_config, 'r') as fd:
                ds_config = json.load(fd)
        
        self.ds_config = self._populate_ds_config(cfg, ds_config)
        logger.info(f"fairseq generated DeepSpeed config: {self.ds_config}")

        self._build_optimizer()

    def _build_optimizer(self):
        ## get non-moe parameters
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )
        
        
        param_groups = create_moe_param_groups(self.model)
        #params = [{"params" : param_groups[0]}, {"moe" : True , "params" : param_groups[1]}]
        #logger.info(params)
        
        #opt_settings = {'lr': 0.0005, 'bias_correction': True, 'betas': (0.9, 0.98), 'eps': 1e-08, 'weight_decay': 0.0, 'max_grad_norm': 0.0}
        #for group in param_groups:
            #group.update(opt_settings)
        
        optimizer = optim.build_optimizer(self.cfg.optimizer, param_groups, ds = True)

        
        #optimizer.param_groups[:] = list(param_groups) + optimizer.param_groups[1:]
       # os.environ['LOCAL_RANK'] = str(self.cfg.distributed_training.device_id)
       # os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = str(self.cfg.distributed_training.device_id)
        self.device = torch.device("cuda", self.cfg.distributed_training.device_id)
        self.model.to(device=self.device)
        
        #logger.info("pg2")
        #logger.info(optimizer.param_groups)
        
        
        engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer._optimizer,
            config_params=self.ds_config
        )
    

        self.zero_enabled = engine.zero_optimization_stage() > 0

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.cfg.lr_scheduler,
            engine.optimizer,
        )
        optimizer.loss_scaler.raise_error_at_min_scale = False
        self._lr_scheduler.step_update(0)
        self._optimizer = optimizer
        self._wrapped_model = engine
        self.device = engine.device
        self._criterion.to(device=self.device)
        torch.distributed.barrier()
        

        if getattr(self.cfg.common, "fp16_scale_window", None) is None:
            if len(self.cfg.optimization.update_freq) > 1:
                raise ValueError(
                    "--fp16-scale-window must be given explicitly when using a "
                    "custom --update-freq schedule"
                )
            data_parallel_size = int(
                self.cfg.distributed_training.distributed_world_size
                / self.cfg.common.model_parallel_size
            )
            scale_window = int(
                2 ** 14 / data_parallel_size / self.cfg.optimization.update_freq[0]
            )
        else:
            scale_window = self.cfg.common.fp16_scale_window

        self.scaler = DynamicLossScaler(
            init_scale=self.cfg.common.fp16_init_scale,
            scale_window=scale_window,
            tolerance=self.cfg.common.fp16_scale_tolerance,
            threshold=self.cfg.common.threshold_loss_scale,
            min_loss_scale=self.cfg.common.min_loss_scale
        )

    @property
    def model(self):
        if self._wrapped_model is None:
            self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def use_distributed_wrapper(self) -> bool:
        return False

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        """DeepSpeed requires save_checkpoint is invoked on all ranks."""
        return True

    @staticmethod
    def _get_config(config, full_name, fairseq_value):
        _config = config
        for name in full_name.split(":"):
            if name in _config:
                _config = _config[name]
            else:
                _config = fairseq_value
                break
        #assert _config == fairseq_value, f"deepspeed config: {full_name} does not align with fairseq value: {fairseq_value}"
        return _config

    def _populate_ds_config(self, cfg, ds_config):
        # gradient accumulation steps
        assert len(self.cfg.optimization.update_freq) == 1, "no support for gradient accumulation schedules"
        ds_config["gradient_accumulation_steps"] = self._get_config(ds_config, "gradient_accumulation_steps", self.cfg.optimization.update_freq[0])

        # train_micro_batch_size_per_gpu
        micro_batch_size = self._get_config(ds_config, "train_micro_batch_size_per_gpu", self.cfg.dataset.max_tokens )
        ds_config["train_micro_batch_size_per_gpu"] = int(micro_batch_size)

        # enable fp16
        fp16 = self._get_config(config=ds_config, full_name="fp16:enabled", fairseq_value=self.cfg.common.fp16)
        if "fp16" not in ds_config:
            ds_config["fp16"] = {}
        ds_config["fp16"]["enabled"] = fp16

        #TODO: patch in fairseq bf16 config

        # gradient_clipping self.cfg.optimization.clip_norm
        ds_config["gradient_clipping"] = self._get_config(ds_config, "gradient_clipping", self.cfg.optimization.clip_norm)

        if "zero_optimization" not in ds_config:
            ds_config["zero_optimization"] = {}

        zero_stage = self._get_config(ds_config, "zero_optimization:stage", cfg.common.zero)
        ds_config["zero_optimization"]["stage"] = zero_stage

        ds_config["zero_allow_untested_optimizer"] = True

        #XXX: force zero elastic checkpoint disabled due to bug
        elastic_ckpt = self._get_config(ds_config, "zero_optimization:elastic_checkpoint", False)
        ds_config["zero_optimization"]["elastic_checkpoint"] = elastic_ckpt

        #XXX: workaround for z3 tracing bug
        ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0

        return ds_config

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        logger.info(f"Saving checkpoint to {filename}")
        
        state_dict = self.state_dict(exclude_model_and_optim=True)
        state_dict["extra_state"].update(extra_state)

        self.model.save_checkpoint(save_dir=filename, client_state=state_dict)

        logger.info(f"Finished saving checkpoint to {filename}")

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        extra_state, self._optim_history, last_optim_state = None, [], None

        logger.info(f"Preparing to load checkpoint {filename}")
        if not os.path.isdir(filename):
            logger.info("No existing checkpoint found {}".format(filename))
            return None

        def load_model(src, dst):
            if torch.distributed.get_rank() == 0:
                print(self.cfg.model)
            dst.load_state_dict(src, strict=False, model_cfg=self.cfg.model)

        load_path, client_states = self.model.load_checkpoint(load_dir=filename, load_optimizer_states=not reset_optimizer, custom_load_fn=load_model)

        logger.info(f'[{torch.distributed.get_rank()}] ckpt client states={client_states}')

        #assert not utils.has_parameters(self.get_criterion()), "criterion w. params not supported yet"
        #extra_state = client_states["extra_state"]

        if not reset_optimizer and not reset_lr_scheduler:
            self.lr_scheduler.load_state_dict(client_states["lr_scheduler_state"])

        self.set_num_updates(client_states["num_updates"])

        self.scaler.loss_scale = client_states["loss_scale"]

        if extra_state is not None:
            itr_state = extra_state["train_iterator"]
            epoch = itr_state["epoch"]

            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            self.lr_step(epoch)

            if itr_state.get("version", 1) >= 2 and itr_state["iterations_in_epoch"] == 0:
                # reset meters at start of epoch
                reset_meters = True

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            logger.info(
                "Loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )
        return extra_state

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=0)

        # If EMA is enabled through store_ema=True
        # and task.uses_ema is True, pass the EMA model as a keyword
        # argument to the task.
        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()

        grad_norm = torch.tensor(0.0).cuda()

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0
        sample_count = len(samples)
        for i, sample in enumerate(samples):
            sample, is_dummy_batch = self._prepare_sample(sample)

            self.model.optimizer.override_loss_scale(self.scaler.loss_scale)

            try:
                # forward and backward
                loss, sample_size_i, logging_output = self.task.train_step(
                    sample=sample,
                    model=self.model,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    update_num=self.get_num_updates(),
                    ignore_grad=is_dummy_batch,
                    **extra_kwargs,
                )
                self.train_step_count += 1
                
                # increment deepspeed micro step on non-final train step since optimizer.step will increment it for us
                if (i + 1) != sample_count:
                    self.model.micro_steps += 1

                if torch.distributed.get_rank() == 0 and self.train_step_count % 25 == 0:
                    logger.info(f"[{torch.distributed.get_rank()}], " \
                        f"micro_step={self.model.micro_steps}, " \
                        f"gas_boundary={self.model.is_gradient_accumulation_boundary()}, " \
                        f"train_step={self.train_step_count}, " \
                        f"lr={self.get_lr()}, " \
                        f"loss_scale={self.model.optimizer.loss_scale}, " \
                        f"loss={loss}")
                del loss
                
                #if self.cfg.common.exit_interval and self.train_step_count % self.cfg.common.exit_interval == 0:
                #    if torch.distributed.get_rank() == 0:
                #        logger.info("exiting early...")
                #    exit()

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
                    if self.cfg.distributed_training.distributed_world_size == 1:
                        return None
                else:
                    raise e

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            train_time = self._local_cumulative_training_time()
            logging_outputs, (
                sample_size,
                ooms,
                total_train_time,
            ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ooms, train_time, ignore=is_dummy_batch
            )
            self._cumulative_training_time = (
                total_train_time / self.data_parallel_world_size
            )

        overflow = False
        try:
            with torch.autograd.profiler.record_function("optimizer"):
                # take an optimization step
                self.task.optimizer_step(
                    self.optimizer, model=self.model, update_num=self.get_num_updates()
                )
                # pass overflow flag from ds to fairseq
                overflow = self.model.optimizer.overflow
                self.scaler.check_overflow(overflow=overflow)
                self.scaler.update()
        except FloatingPointError:
            raise
        except OverflowError as e:
            logger.info(f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}")
            overflow = True
    
        logging_output = None
        if not overflow or self.cfg.distributed_training.ddp_backend == "slow_mo":
            self.set_num_updates(self.get_num_updates() + 1)

            if self.cuda and self.cuda_env is not None:
                # log minimum free memory over the iteration
                gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
                gb_free = self.cuda_env.total_memory_in_GB - gb_used
                metrics.log_scalar(
                    "gb_free", gb_free, priority=1500, round=1, weight=0
                )

            # log stats
            logging_output = self._reduce_and_log_stats(
                logging_outputs, sample_size, grad_norm
            )

        if self.cfg.common.fp16:
            metrics.log_scalar(
                "loss_scale",
                self.optimizer.cur_scale,
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output