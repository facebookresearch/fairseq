# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import subprocess
import typing as tp
from collections import defaultdict
from dataclasses import dataclass
from random import randint

import hydra
import stopes.core
from omegaconf import MISSING, DictConfig
from stopes.core.launcher import NoCache


@dataclass
class ClusterConfig:
    name: str = MISSING
    data_dir: str = MISSING
    partition: str = MISSING
    constraint: str = ""
    flores_path: str = MISSING
    memory_multiplier: int = 0
    timeout_min: int = 1000


@dataclass
class LangConfig:
    bin_root: str = MISSING
    all: tp.List[str] = MISSING
    sampled: tp.List[str] = MISSING


@dataclass
class ModelConfig:
    langs: tp.List[str] = MISSING


@dataclass
class GenerateMultiConfig:
    model_type: str = "dense"
    direction: str = "en_to_many"
    # use group: example
    cluster: ClusterConfig = ClusterConfig()
    lang_config: LangConfig = LangConfig()
    model_config: ModelConfig = ModelConfig()
    # all or sample
    eval_on: str = "sampled"
    lang_pairs_per_job: int = 1
    model_folder: tp.List[str] = MISSING
    checkpoints: tp.List[str] = MISSING
    gen_splits: tp.List[str] = MISSING
    spm_model: str = MISSING
    data: str = MISSING
    encoder_langtok: str = "tgt"
    output_dir: str = MISSING
    beam_size: int = 4
    fp16: bool = False
    fairseq_root: str = "."
    metrics_only: bool = False
    replication_count: int = 1
    finetune_dict_specs: tp.Optional[str] = None
    add_data_source_prefix_tags: bool = False
    moe_eval_cap: float = 1.0
    datalabel: str = ""


@dataclass
class JobConfig:
    gen_split: str
    checkpoint: str
    lang_pairs: tp.List[str]
    datalabel: str = ""


class GenerateMultiModule(stopes.core.StopesModule):
    def __init__(self, config):
        super().__init__(config)
        assert os.path.isdir(
            config.model_folder
        ), f"{config.model_folder} is not a directory"
        assert os.access(
            config.model_folder, os.R_OK
        ), f"{config.model_folder} is not a readable"

    def array(self):
        def chunk(lst, chunk_size):
            chunks = []
            i = 0
            while i < len(lst):
                chunks.append(lst[i : min(i + chunk_size, len(lst))])
                i = i + chunk_size
            print(chunks)
            return chunks

        lang_pairs = (
            self.config.lang_config.all
            if self.config.eval_on == "all"
            else self.config.lang_config.sampled
        )
        filtered_pairs = []
        for lang_pair in lang_pairs:
            if (
                self.config.direction == "all"
                or (
                    self.config.direction == "en_to_many"
                    and lang_pair.startswith("eng-")
                )
                or (
                    self.config.direction == "many_to_en" and lang_pair.endswith("-eng")
                )
                or (self.config.direction == "non_english" and "eng" not in lang_pair)
            ):
                filtered_pairs.append(lang_pair)
        lang_pairs_chunks = chunk(filtered_pairs, self.config.lang_pairs_per_job)
        return [
            JobConfig(
                gen_split=split,
                checkpoint=chk,
                lang_pairs=lang_pairs,
                datalabel=self.config.datalabel,
            )
            for split in self.config.gen_splits
            for chk in self.config.checkpoints
            for lang_pairs in lang_pairs_chunks
        ]

    def requirements(self):
        if self.config.model_type == "moe":
            gpus = 8
            num_nodes = 2
            req = stopes.core.DistributedRequirements(
                tasks_per_node=1,
                nodes=num_nodes,
                gpus_per_node=gpus,
                cpus_per_task=gpus * 10,
                mem_gb=gpus * self.config.cluster.memory_multiplier,
                timeout_min=self.config.cluster.timeout_min,
                constraint=self.config.cluster.constraint,
            )
            return req
        else:
            return stopes.core.DistributedRequirements(
                tasks_per_node=1,
                nodes=1,
                gpus_per_node=8,
                cpus_per_task=8,
                mem_gb=48,
                timeout_min=self.config.cluster.timeout_min,
                constraint=self.config.cluster.constraint,
            )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        job_config = iteration_value
        lang_pairs = job_config.lang_pairs
        for lang_pair in lang_pairs:
            try:
                src = lang_pair.split("-")[0]
                tgt = lang_pair.split("-")[1]
                if self.config.finetune_dict_specs is not None:
                    finetune_dict_specs = (
                        f' --finetune-dict-specs "{self.config.finetune_dict_specs}"'
                    )
                else:
                    finetune_dict_specs = ""
                if self.config.model_type == "moe":
                    max_sentences = 36
                    cap = self.config.moe_eval_cap
                    req = self.requirements()
                    world_size = req.nodes * req.gpus_per_node
                    port = (randint(0, 32767) % 119) + 15_000
                    model_overrides = {
                        "world_size": world_size,
                        "moe_eval_capacity_token_fraction": cap,
                        "use_moe_pad_mask": False,
                        "pass_tokens_transformer_layer": False,
                        "replication_count": self.config.replication_count,
                    }
                    moe_params = (
                        "--is-moe "
                        f"--distributed-world-size {world_size} "
                        f"--distributed-port {port} "
                        f'--model-overrides "{repr(model_overrides)}" '
                    )
                else:
                    moe_params = ""
                    max_sentences = 50
                    cap = "no_cap"

                if job_config.datalabel:
                    out_dir = os.path.join(
                        self.config.output_dir,
                        f"gen_output_{job_config.datalabel}_{cap}",
                        f"{src}-{tgt}_{job_config.checkpoint}_{job_config.gen_split}",
                    )
                else:
                    out_dir = os.path.join(
                        self.config.output_dir,
                        f"gen_output_{cap}",
                        f"{src}-{tgt}_{job_config.checkpoint}_{job_config.gen_split}",
                    )

                os.makedirs(out_dir, exist_ok=True)
                # check if generation was completed before
                has_completed_before = False
                if os.path.exists(os.path.join(out_dir, "bleu.results")):
                    with open(os.path.join(out_dir, "bleu.results"), "r") as f:
                        for line in f.readlines():
                            if "BLEU" in line:
                                has_completed_before = True
                if has_completed_before:
                    continue
                checkpoint_name = job_config.checkpoint
                if self.config.model_type == "dense":
                    checkpoint_name += "-shard0"
                model = os.path.join(
                    self.config.model_folder, f"{job_config.checkpoint}.pt"
                )
                # TODO maybe call generate main directly here with a hydra config
                generate_command = (
                    f"python {self.config.fairseq_root}/fairseq_cli/generate.py "
                    f" {self.config.data} "
                    f" --path {model} "
                    f" --task translation_multi_simple_epoch "
                    f" --langs \"{','.join(self.config.model_config.langs)}\" "
                    f'--lang-pairs "{src}-{tgt}"'
                    f" --source-lang {src} "
                    f" --target-lang {tgt} "
                    f'--encoder-langtok "{self.config.encoder_langtok}"'
                    " --decoder-langtok "
                    f" --gen-subset {job_config.gen_split} "
                    f" --beam {self.config.beam_size} "
                    " --bpe 'sentencepiece' "
                    f" --sentencepiece-model {self.config.spm_model} "
                    " --sacrebleu "
                    " --enable-m2m-validation "
                    f"{'--add-data-source-prefix-tags' if self.config.add_data_source_prefix_tags else ''}"
                    f" {'--fp16' if self.config.fp16 else ''}"
                    f" {moe_params} "
                    f" {finetune_dict_specs} "
                    f" --max-sentences {max_sentences} "
                    f" --results-path {out_dir} >> {out_dir}/eval.out 2>&1 "
                )
                post_proc_command = (
                    f"/bin/bash -o pipefail -c '"
                    + f"cat {out_dir}/generate-{job_config.gen_split}.txt"
                    + ' | grep -aP "^D-"'
                    + " | sort -nr -k1.2 "
                    + " | cut -f3' "
                    + " | sed 's/^<MINED_DATA> //g' "
                    + f" > {out_dir}/gen_best.output"
                )
                if job_config.datalabel:
                    ref_split = "dev" if job_config.gen_split == "valid" else "test"
                    ref_file = os.path.join(
                        self.config.cluster.non_flores_path,
                        job_config.datalabel,
                        f"{ref_split}.{src}-{tgt}.{tgt}",
                    )
                else:
                    # Default evaluation on Flores
                    flores_split = (
                        "dev" if job_config.gen_split == "valid" else "devtest"
                    )
                    ref_file = os.path.join(
                        self.config.cluster.flores_path,
                        flores_split,
                        f"{tgt}.{flores_split}",
                    )
                # Install `pip install git+https://github.com/mjpost/sacrebleu.git@master`
                # spm BLEU Eval
                bleu_command = (
                    f"SACREBLEU_FORMAT=text "
                    + f"sacrebleu -tok spm {ref_file} < {out_dir}/gen_best.output "
                    + f" > {out_dir}/bleu.results"
                )
                # chrf++ Eval
                chrf_command = (
                    f"sacrebleu -m chrf --chrf-word-order 2 -tok spm {ref_file} < {out_dir}/gen_best.output "
                    + f" > {out_dir}/chrf.results"
                )
                if job_config.datalabel:
                    # --tok intl (--zh for zh)
                    sacrebleu_command = (
                        f"SACREBLEU_FORMAT=text "
                        + f"sacrebleu -tok {'zh' if tgt == 'zho_Hans' else 'intl'}"
                        + f" {ref_file} < {out_dir}/gen_best.output "
                        + f" > {out_dir}/sacrebleu.results"
                    )
                else:
                    sacrebleu_command = ""

                full_command = "\n".join(
                    [
                        generate_command,
                        post_proc_command,
                        bleu_command,
                        chrf_command,
                        sacrebleu_command,
                    ]
                )
                if self.config.get("debug", False):
                    print(full_command)
                else:
                    generate_command_file = os.path.join(out_dir, "gen.sh")
                    with open(generate_command_file, "w") as f:
                        f.write(full_command)
                    subprocess.run(
                        full_command,
                        shell=True,
                        check=True,
                    )
            except Exception as e:
                print(e)
                continue


def get_type(pair):
    if "-" not in pair:
        return None
    from examples.nllb.modeling.evaluation.train_example_count import flores200_public
    from examples.nllb.modeling.evaluation.train_example_count.code_mapping import (
        lang_code_map,
    )

    train_counts2 = flores200_public.train_counts
    # High resource +1M, low resource 0-1M; Very low resource <0.1M
    low_limits = {"high": 1000000, "low": 0, "v_low": 0}
    high_limits = {"high": 10000000000, "low": 1000000, "v_low": 100000}
    src, tgt = pair.split("-")
    if src not in train_counts2 or tgt not in train_counts2:
        if src in lang_code_map:
            src = lang_code_map[src]
        if tgt in lang_code_map:
            tgt = lang_code_map[tgt]
        if src not in train_counts2 or tgt not in train_counts2:
            print(f"{src} or {tgt} is not in train_counts")
            return None
    count = min(train_counts2[src], train_counts2[tgt])
    resource = None
    for t in low_limits.keys():
        if count >= low_limits[t] and count <= high_limits[t]:
            resource = t
            break
    vlow_res = None
    if count >= low_limits["v_low"] and count <= high_limits["v_low"]:
        vlow_res = "v_low"

    return resource, vlow_res


def get_averages(scores_map, threshold=0):
    en_xx = defaultdict(list)
    xx_en = defaultdict(list)
    non_eng = defaultdict(list)
    all_pairs = defaultdict(list)
    for pair, score in scores_map.items():
        resource, vlow_res = get_type(pair)
        if score < threshold:
            print(f"{pair} {score} is skipped due to threshold")
            continue
        if resource is None:
            print(f"{pair} {score} is skipped due to missing resource level")
            continue
        all_pairs["all"].append(score)
        all_pairs[resource].append(score)
        if vlow_res is not None:
            all_pairs[vlow_res].append(score)
        if pair.startswith("eng_Latn-"):
            en_xx[resource].append(score)
            if vlow_res is not None:
                en_xx[vlow_res].append(score)
            en_xx["all"].append(score)
        elif pair.endswith("-eng_Latn"):
            xx_en[resource].append(score)
            if vlow_res is not None:
                xx_en[vlow_res].append(score)
            xx_en["all"].append(score)
        else:
            non_eng[resource].append(score)
            if vlow_res is not None:
                non_eng[vlow_res].append(score)
            non_eng["all"].append(score)
    avg_en_xx = defaultdict(int)
    avg_xx_en = defaultdict(int)
    avg_non_eng = defaultdict(int)
    avg_all_pairs = defaultdict(int)
    lists = [en_xx, xx_en, non_eng, all_pairs]
    averages = [avg_en_xx, avg_xx_en, avg_non_eng, avg_all_pairs]
    for idx, agg in enumerate(averages):
        lst = lists[idx]
        for resource in ["all", "high", "low", "v_low"]:
            agg[resource] = round(sum(lst[resource]) / max(len(lst[resource]), 1), 2)
    return {
        "en-xx": avg_en_xx,
        "xx-en": avg_xx_en,
        "non-eng": avg_non_eng,
        "all": avg_all_pairs,
    }


async def tabulate(config: DictConfig) -> None:
    if config.model_type == "moe":
        cap = config.moe_eval_cap
    else:
        cap = "no_cap"
    if config.datalabel:
        out_dir = os.path.join(
            config.output_dir,
            f"gen_output_{config.datalabel}_{cap}",
        )
    else:
        out_dir = os.path.join(
            config.output_dir,
            f"gen_output_{cap}",
        )

    out_dir = os.path.join(config.output_dir, f"gen_output_{cap}")
    lang_pairs = config.lang_config.all

    for chk in config.checkpoints:
        for split in config.gen_splits:
            bleus_map = {}
            chrf_map = {}
            lang_pairs = sorted(list(set(lang_pairs)))
            for lang_pair in lang_pairs:
                src, tgt = lang_pair.split("-")
                pair_dir = f"{src}-{tgt}_{chk}_{split}"
                bleu_fpath = os.path.join(out_dir, pair_dir, "bleu.results")
                command = f"cat {bleu_fpath} | cut -d' ' -f3 "
                bleu_str = subprocess.check_output(command, shell=True).decode()
                bleu = round(float(bleu_str), 2) if len(bleu_str) > 0 else -1
                bleus_map[lang_pair] = bleu
                chrf_fpath = os.path.join(out_dir, pair_dir, "chrf.results")
                command = f"grep score {chrf_fpath} | cut -f3 -d' ' | cut -f1 -d','"
                chrf_str = subprocess.check_output(command, shell=True).decode()
                chrf = round(float(chrf_str), 2) if len(chrf_str) > 0 else -1
                chrf_map[lang_pair] = chrf
            average_bleus = get_averages(bleus_map)
            average_chrfs = get_averages(chrf_map)
            for metric in ["bleu", "chrf"]:
                output_fpath = os.path.join(out_dir, f"{metric}_{chk}_{split}.tsv")
                with open(output_fpath, "w") as fout:
                    average_vals = average_bleus if metric == "bleu" else average_chrfs
                    metric_map = bleus_map if metric == "bleu" else chrf_map
                    for subset, dict_values in average_vals.items():
                        for k, v in dict_values.items():
                            print(f"{subset}_{k}\t{v}", file=fout)
                    for pair in lang_pairs:
                        print(f"{pair}\t{metric_map[pair]}", file=fout)


async def _main(config):
    if not config.metrics_only:
        launcher = hydra.utils.instantiate(config.launcher)
        module = GenerateMultiModule(config)
        await launcher.schedule(module)
    # tabulate results
    await tabulate(config)


@hydra.main(config_path="conf", config_name="generate_multi")
def main(config: DictConfig) -> None:
    asyncio.run(_main(config))


if __name__ == "__main__":
    main()
