# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import subprocess
import typing as tp
from dataclasses import dataclass
from random import randint
from re import M

import hydra
from omegaconf import MISSING, DictConfig, OmegaConf
from stopes.core import StopesModule


@dataclass
class ClusterConfig:
    cluster_name: str = MISSING
    partition: str = MISSING
    memory_multiplier: int = 0
    timeout_min: int = 1000


@dataclass
class DatasetConfig:
    dataset_name: str = MISSING
    num_shards: int = MISSING
    langs: str = MISSING
    langs_file: str = MISSING
    lang_pairs: str = MISSING
    lang_pairs_file: str = MISSING
    eval_lang_pairs: str = MISSING
    eval_lang_pairs_file: str = MISSING
    data_prefix: tp.Dict[str, str] = MISSING


@dataclass
class ModelTypeConfig:
    name: str = MISSING
    moe_params: str = MISSING
    expert_count: int = MISSING


@dataclass
class TrainConfig:
    cluster: ClusterConfig = ClusterConfig()
    dataset: DatasetConfig = DatasetConfig()
    model_type: ModelTypeConfig = ModelTypeConfig()
    fairseq_root: str = MISSING
    output_dir: str = MISSING
    log_dir: str = None
    train_prefix: str = MISSING
    seed: int = MISSING
    arch: str = MISSING
    max_updates: int = MISSING
    max_update_str: str = None
    resume_finished: bool = False
    synchronize_checkpoints_before_copy: bool = False
    validate_interval_updates: int = MISSING
    keep_interval_updates: int = 10
    symlink_best_and_last_checkpoints: bool = False
    save_interval_updates: int = MISSING
    save_interval: int = 1
    best_checkpoint_metric: str = MISSING
    encoder_langtok: str = MISSING
    ddp_backend: str = MISSING
    fp16: bool = MISSING
    lr: float = MISSING
    warmup: int = MISSING
    max_tokens: int = MISSING
    update_freq: int = MISSING
    num_nodes: int = MISSING
    num_gpus_per_node: int = MISSING
    temp: float = MISSING
    dropout: float = MISSING
    module_name: str = "examples.nllb.modeling.sweep.sweep_mmt"
    num_trials: int = 1
    max_time_mins: int = 4320
    mem: int = 0
    moe_eval_cap: float = 1.0
    checkpoint_activations: bool = False
    zero2: bool = False
    train_subset: str = "train"
    ssl_task: str = None
    dae_mask: float = 0.3
    finetune_dict_specs: tp.Optional[str] = None
    restore_file: tp.Optional[str] = None
    finetune_from_model: tp.Optional[str] = None
    no_save: bool = False
    log_interval: int = 100
    reset_dataloader: bool = False
    reset_all: bool = False
    replication_count: int = 1


@dataclass
class MainConfig:
    cfg: TrainConfig = TrainConfig()


class TrainModule(StopesModule):
    def __init__(self, config):
        super().__init__(config)

        # values in cfg configurable through entire .yaml files in conf/cfg/
        cfg = config.cfg

        self.output_dir = cfg.output_dir
        self.log_dir = cfg.log_dir or cfg.output_dir
        os.makedirs(self.log_dir, exist_ok=True)
        print("TRAINING DIR: ", self.log_dir)

        config_yaml_file = os.path.join(self.log_dir, "train_script.yaml")
        with open(config_yaml_file, "w") as f:
            f.write(OmegaConf.to_yaml(config.cfg))

        cluster_name = cfg.cluster.cluster_name
        assert cluster_name in cfg.dataset.data_prefix
        data_prefix = cfg.dataset.data_prefix[cluster_name]
        assert data_prefix is not None
        assert os.path.isdir(data_prefix), f"{data_prefix} is not a directory"
        assert os.access(data_prefix, os.R_OK), f"{data_prefix} is not readable"

        data_dir = ""
        for shard_id in range(cfg.dataset.num_shards):
            data_dir += f":{data_prefix}/shard{shard_id:03d}"
        data_dir = data_dir[1:]  # remove leading colon
        print("data_dir: ", data_dir)
        self.data_dir = data_dir

    def launch_job(self):
        # values in cfg configurable through entire .yaml files in conf/cfg/
        cfg = self.config.cfg
        log_dir_str = (
            f"--log-dir {self.log_dir} --skip-create-save-dir" if cfg.log_dir else ""
        )
        max_update_str = (
            f"--max-update-str {cfg.max_update_str}" if cfg.max_update_str else ""
        )

        if cfg.dataset.langs is None:
            assert cfg.dataset.langs_file is not None
            langs = os.path.join(cfg.fairseq_root, cfg.dataset.langs_file)
        else:
            langs = cfg.dataset.langs

        if cfg.dataset.lang_pairs is None:
            assert cfg.dataset.lang_pairs_file is not None
            lang_pairs = os.path.join(cfg.fairseq_root, cfg.dataset.lang_pairs_file)
        else:
            lang_pairs = cfg.dataset.lang_pairs

        if cfg.dataset.eval_lang_pairs is not None:
            eval_lang_pairs = f"--eval-lang-pairs {cfg.dataset.eval_lang_pairs}"
        elif cfg.dataset.eval_lang_pairs_file is not None:
            eval_lang_pairs = f"--eval-lang-pairs {os.path.join(cfg.fairseq_root, cfg.dataset.eval_lang_pairs_file)}"
        else:
            eval_lang_pairs = ""

        tensorboard_dir = os.path.join(self.output_dir, "tb")

        checkpoint_activations_param = (
            "--checkpoint-activations" if cfg.checkpoint_activations else ""
        )
        zero2_param = "--zero2" if cfg.zero2 else ""

        print("MoE params ", cfg.model_type.moe_param)
        if cfg.model_type.moe_param:
            moe_params = (
                cfg.model_type.moe_param
                + f" --moe-expert-count {cfg.model_type.expert_count}"
            )
        else:
            moe_params = ""

        ssl_params = ""
        if getattr(cfg, "ssl_task", None):
            assert getattr(cfg.dataset, "mono_num_shards", None) is not None
            assert getattr(cfg.dataset, "mono_data_prefix", None) is not None
            assert getattr(cfg.dataset, "mono_langs", None) is not None

            cluster_name = cfg.cluster.cluster_name
            assert cluster_name in cfg.dataset.mono_data_prefix
            mono_data_prefix = cfg.dataset.mono_data_prefix[cluster_name]

            mono_data_dir = ""
            for shard_id in range(cfg.dataset.mono_num_shards):
                mono_data_dir += f":{mono_data_prefix}/shard{shard_id:03d}"
            mono_data_dir = mono_data_dir[1:]  # remove leading colon
            print("mono_data_dir: ", mono_data_dir)
            mono_langs = cfg.dataset.mono_langs

            dae_mask = getattr(cfg, "dae_mask", 0.3)
            ssl_params = f"""--extra-data "{{'{cfg.ssl_task}': '{mono_data_dir}'}}" \
                --extra-lang-pairs "{{'{cfg.ssl_task}': '{mono_langs}'}}" \
                --langtoks "{{'{cfg.ssl_task}': ('src', 'tgt')}}"  \
                --mask-length span-poisson \
                --mask {dae_mask} \
                --mask-random 0.1 \
                --poisson-lambda 3.5"""

        sweep_command = f"""
            cd {cfg.fairseq_root}
            python -m {cfg.module_name} \
                -d {self.data_dir} \
                -p {cfg.train_prefix} \
                --checkpoints-dir {self.output_dir} \
                {log_dir_str} \
                --partition {cfg.cluster.partition} \
                -t {cfg.num_trials} \
                -n {cfg.num_nodes} \
                -g {cfg.num_gpus_per_node} \
                --resume-failed \
                --time {cfg.max_time_mins} \
                --mem {cfg.mem} \
                --sampling-method temperature \
                --sampling-temperature {cfg.temp} \
                --decoder-langtok \
                --encoder-langtok {cfg.encoder_langtok} \
                --langs {langs} \
                --lang-pairs {lang_pairs} \
                {eval_lang_pairs} \
                --moe-eval-cap {cfg.moe_eval_cap} \
                --ddp-backend {cfg.ddp_backend} \
                --max-update {cfg.max_updates} \
                {max_update_str} \
                {"--resume-finished" if cfg.resume_finished else ""} \
                {"--synchronize-checkpoints-before-copy" if cfg.synchronize_checkpoints_before_copy else ""} \
                --max-tokens {cfg.max_tokens} \
                --update-freq {cfg.update_freq} \
                --warmup-updates {cfg.warmup} \
                --lr {cfg.lr} \
                --opt adam16bit \
                --share-all-embeddings \
                --save-interval-updates {cfg.save_interval_updates} \
                --save-interval {cfg.save_interval} \
                --tensorboard-logdir {tensorboard_dir} \
                --arch {cfg.arch} \
                --dropout {cfg.dropout} \
                --validate-interval-updates {cfg.validate_interval_updates} \
                --keep-interval-updates {cfg.keep_interval_updates} \
                {"--symlink-best-and-last-checkpoints" if cfg.symlink_best_and_last_checkpoints else ""} \
                --best-checkpoint-metric {cfg.best_checkpoint_metric} \
                --seed {cfg.seed} \
                --train-subset {cfg.train_subset} \
                --snapshot-code \
                --use-local-shard-size \
                --enable-m2m-validation \
                --add-data-source-prefix-tags \
                --replication-count {cfg.replication_count} \
                {checkpoint_activations_param} \
                {zero2_param} \
                {moe_params} \
                {ssl_params} \
                {f"--finetune-dict-specs {cfg.finetune_dict_specs} " if cfg.finetune_dict_specs is not None else ""} \
                {f"--restore-file {cfg.restore_file}" if cfg.restore_file is not None else ""} \
                {f"--finetune-from-model {cfg.finetune_from_model}" if cfg.finetune_from_model is not None else ""} \
                {"--no-save" if cfg.no_save else ""} \
                {f"--log-interval {cfg.log_interval}" if cfg.log_interval is not None else ""} \
                {f"--eval-lang-pairs {cfg.eval_lang_pairs}" if cfg.eval_lang_pairs is not None else ""} \
                {"--reset-dataloader" if cfg.reset_dataloader else ""} \
                {"--reset-all" if cfg.reset_all else ""}
        """

        print("RUNNING SWEEP COMMAND:")
        print(sweep_command)

        subprocess.run(
            sweep_command,
            shell=True,
            check=True,
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        # launching one training job synchronously for now
        pass


@hydra.main(config_path="conf", config_name="base_config")
def main(config: DictConfig) -> None:
    train_module = TrainModule(config)
    train_module.launch_job()


if __name__ == "__main__":
    main()
