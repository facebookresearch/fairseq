# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import scipy

import torch
import torch.multiprocessing as mp
from fairseq import checkpoint_utils, options
from fairseq.data.codedataset import CodeDataset, ExpressiveCodeDataConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from torch.utils.data import DataLoader, DistributedSampler
from fairseq.utils import move_to_cuda
from fairseq import utils
from fairseq.criterions.speech_ulm_criterion import nll_loss, mae_loss

import time
from types import SimpleNamespace

import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from naive_decoder import Naive_F0_Decoder
from inference_dataset import InferenceDataset, explode_batch
from sample.sample import do_sampling, TemperatureDecoder, FilterNamesDataset

try:
    from nltk.translate.bleu_score import sentence_bleu
except ImportError:
    print("Please install nltk: `pip install --user -U nltk`")
    raise


@torch.no_grad()
def teacher_force_everything(
    args, dataset, model, criterion, tgt_dict, rank, world_size
):
    prefix = args.prefix_length

    f0_decoder = None
    if args.dequantize_prosody:
        assert dataset.discrete_f0
        print("Reporting MAE for a discrete model")
        f0_decoder = Naive_F0_Decoder(
            args.f0_discretization_bounds, dataset.config.f0_vq_n_units
        ).cuda()

    dataset = InferenceDataset(
        dataset,
        prefix=args.prefix_length,
        only_prefix=False,
        filter_short=True,
        presort_by_length=True,
    )
    sampler = (
        None
        if world_size == 1
        else DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    )
    dataloader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=False,
        collate_fn=dataset.collater,
        sampler=sampler,
    )

    total_token_loss, total_duration_loss, total_f0_loss, total_tokens = (
        0.0,
        0.0,
        0.0,
        0.0,
    )

    i = 0
    for batch in dataloader:
        i += 1
        batch = move_to_cuda(batch)
        output = model(**batch["net_input"])

        tokens, durations, f0 = output["token"], output["duration"], output["f0"]
        durations, f0 = durations.squeeze(), f0.squeeze()

        token_loss = nll_loss(
            tokens[:, prefix - 1 :],
            batch["target"][:, prefix - 1 :].contiguous(),
            batch["mask"][:, prefix - 1 :].contiguous(),
            reduce=True,
        )

        if args.dequantize_prosody:
            durations = durations.argmax(dim=-1)
            duration_loss = mae_loss(
                durations[:, prefix - 1 :].contiguous().float(),
                batch["dur_target"][:, prefix - 1 :].contiguous().float(),
                batch["dur_mask"][:, prefix - 1 :].contiguous(),
                reduce=True,
            )
        else:
            duration_loss = criterion.dur_loss_fn(
                durations[:, prefix - 1 :].contiguous(),
                batch["dur_target"][:, prefix - 1 :].contiguous(),
                batch["dur_mask"][:, prefix - 1 :].contiguous(),
                reduce=True,
            )

        if f0_decoder:
            f0 = f0.argmax(dim=-1)
            f0 = f0_decoder(f0).squeeze(-1)

            f0_target = batch["raw_f0"]
            f0_loss = mae_loss(
                f0[:, prefix - 1 :].contiguous(),
                f0_target[:, prefix - 1 :].contiguous(),
                batch["f0_mask"][:, prefix - 1 :].contiguous(),
                reduce=True,
            )
        else:
            f0_loss = criterion.f0_loss_fn(
                f0[:, prefix - 1 :].contiguous(),
                batch["f0_target"][:, prefix - 1 :].contiguous(),
                batch["f0_mask"][:, prefix - 1 :].contiguous(),
                reduce=True,
            )

        n_tokens = (~batch["dur_mask"])[:, prefix - 1 :].sum()

        total_token_loss += token_loss.item()
        total_duration_loss += duration_loss.item()
        total_f0_loss += f0_loss.item()

        total_tokens += n_tokens.item()
        if args.debug and i > 5:
            break

    values = torch.tensor([total_token_loss, total_duration_loss, total_f0_loss])
    normalizers = torch.tensor([total_tokens for _ in range(3)])

    return values, normalizers


def get_bleu(produced_tokens, target_tokens, tgt_dict):
    assert target_tokens.ndim == 1
    assert produced_tokens.size(1) == target_tokens.size(0)

    # we can have padding due to shifted channels
    shift = 0
    for token in reversed(target_tokens.cpu().tolist()):
        if token in [tgt_dict.pad(), tgt_dict.eos()]:
            shift += 1
        else:
            break
    target_tokens = target_tokens[:-shift]
    produced_tokens = produced_tokens[:, :-shift]

    string_target = tgt_dict.string(target_tokens).split()
    string_candidates = [
        tgt_dict.string(produced_tokens[i, :]).split()
        for i in range(produced_tokens.size(0))
    ]

    bleu3 = sentence_bleu(
        references=string_candidates,
        hypothesis=string_target,
        weights=(1.0 / 3, 1.0 / 3, 1.0 / 3),
    )
    return bleu3


@torch.no_grad()
def continuation(args, dataset, model, criterion, tgt_dict, rank, world_size):
    is_discrete_duration = dataset.discrete_dur
    is_discrete_f0 = dataset.discrete_f0

    f0_decoder = None
    if args.dequantize_prosody:
        assert dataset.discrete_f0
        print("Reporting MAE F0 for a discrete model")
        f0_decoder = Naive_F0_Decoder(
            args.f0_discretization_bounds, dataset.config.f0_vq_n_units
        ).cuda()

    dataset = InferenceDataset(
        dataset, args.prefix_length, filter_short=True, presort_by_length=True
    )
    sampler = (
        None
        if world_size == 1
        else DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collater,
        sampler=sampler,
    )

    Ts = args.T_token, args.T_duration, args.T_f0
    decoder = TemperatureDecoder(
        Ts, discrete_dur=is_discrete_duration, discrete_f0=is_discrete_f0
    )

    running_stats = SimpleNamespace(
        token_bleu=0.0,
        duration_nll=0.0,
        duration_mae=0.0,
        f0_nll=0.0,
        f0_mae=0.0,
        n_tokens=0.0,
        n_sentences=0.0,
        f0_sum=0.0,
        f0_sum_sq=0.0,
        dur_sum=0.0,
        dur_sum_sq=0.0,
    )

    for i, batch in enumerate(dataloader):
        batch = explode_batch(batch, args.batch_explosion_rate)
        bsz = batch["target"].size(0)

        batch = move_to_cuda(batch)
        prefix = batch["prefix"][0]

        max_length_to_unroll = batch["target"].size(1)
        prefix_length = batch["net_input"]["src_tokens"].size(1)
        steps = max_length_to_unroll - prefix_length + 1

        assert steps > 0
        produced_tokens, produced_durations, produced_f0, outputs = do_sampling(
            model,
            batch,
            tgt_dict.eos(),
            decoder,
            autoregressive_steps=steps,
            teacher_force_tokens=args.teacher_force_tokens,
            teacher_force_duration=args.teacher_force_duration,
            teacher_force_f0=args.teacher_force_f0,
        )

        if args.teacher_force_tokens:
            assert (produced_tokens[:, 1:] == batch["target"]).all()
        if args.teacher_force_duration:
            assert (produced_durations[:, 1:] == batch["dur_target"]).all()
        if args.teacher_force_f0:
            assert (produced_f0[:, 1:] == batch["f0_target"]).all()

        dur_target = batch["dur_target"][:, prefix - 1 :].contiguous()
        f0_target = batch["f0_target"][:, prefix - 1 :].contiguous()

        f0_mask = batch["f0_mask"][:, prefix - 1 :].contiguous()
        dur_mask = batch["dur_mask"][:, prefix - 1 :].contiguous()

        duration_mae = mae_loss(
            produced_durations[:, prefix:].float(),
            dur_target.float(),
            dur_mask,
            reduce=False,
        )
        min_duration_mae = duration_mae.view(bsz, -1).sum(dim=-1).min(dim=0)[0]
        running_stats.duration_mae += min_duration_mae

        running_stats.dur_sum += (
            produced_durations[:, prefix:].float() * (~dur_mask)
        ).sum() / args.batch_explosion_rate
        running_stats.dur_sum_sq += (
            produced_durations[:, prefix:].float() * (~dur_mask)
        ).pow(2.0).sum() / args.batch_explosion_rate

        if is_discrete_duration:
            duration_loss = criterion.dur_loss_fn(
                torch.stack([x[1] for x in outputs], dim=1),
                dur_target,
                dur_mask,
                reduce=False,
            )
            min_duration_loss = duration_loss.view(bsz, -1).sum(dim=-1).min(dim=0)[0]
            running_stats.duration_nll += min_duration_loss

        if f0_decoder:  # can only exist for discrete F0 models
            decoded_produced_f0 = f0_decoder(produced_f0[:, prefix:])
            decoded_f0_target = batch["raw_f0"][:, prefix - 1 :].contiguous()

            if produced_f0.ndim == 3:
                decoded_produced_f0 = decoded_produced_f0.squeeze(2)
                decoded_f0_target = decoded_f0_target.squeeze(2)

            f0_mae = mae_loss(
                decoded_produced_f0, decoded_f0_target, f0_mask, reduce=False
            )
            f0_mae = f0_mae.view(bsz, -1).sum(dim=-1).min(dim=0)[0]
            running_stats.f0_mae += f0_mae

            f0_loss = criterion.f0_loss_fn(
                torch.stack([x[2] for x in outputs], dim=1),
                f0_target.long(),
                f0_mask,
                reduce=False,
            )
            f0_loss = f0_loss.view(bsz, -1).sum(dim=-1).min(dim=0)[0]
            running_stats.f0_nll += f0_loss

            running_stats.f0_sum += (
                decoded_produced_f0 * (~f0_mask)
            ).sum() / args.batch_explosion_rate
            running_stats.f0_sum_sq += (decoded_produced_f0 * (~f0_mask)).pow(
                2.0
            ).sum() / args.batch_explosion_rate

        else:
            assert not is_discrete_duration

            f0_loss = mae_loss(
                produced_f0[:, prefix:], f0_target, f0_mask, reduce=False
            )
            f0_loss = f0_loss.view(bsz, -1).sum(dim=-1).min(dim=0)[0]
            running_stats.f0_mae += f0_loss

            running_stats.f0_sum += (
                produced_f0[:, prefix:].sum() / args.batch_explosion_rate
            )
            running_stats.f0_sum_sq += (
                produced_f0[:, prefix:].pow(2.0).sum() / args.batch_explosion_rate
            )

        running_stats.n_tokens += (~dur_mask)[0, ...].sum()

        token_loss = get_bleu(
            produced_tokens[:, prefix:], batch["target"][0, prefix - 1 :], tgt_dict
        )
        running_stats.token_bleu += token_loss
        running_stats.n_sentences += 1

        if args.debug:
            break

    values = torch.tensor(
        [
            running_stats.token_bleu,
            running_stats.duration_nll,
            running_stats.duration_mae,
            running_stats.f0_nll,
            running_stats.f0_mae,
            running_stats.f0_sum,
            running_stats.f0_sum_sq,
            running_stats.dur_sum,
            running_stats.dur_sum_sq,
        ]
    )
    normalizers = torch.tensor(
        [running_stats.n_sentences] + [running_stats.n_tokens] * 8
    )

    return values, normalizers


@torch.no_grad()
def correlation(args, dataset, model, criterion, tgt_dict, rank, world_size):
    is_discrete_duration = dataset.discrete_dur
    is_discrete_f0 = dataset.discrete_f0

    f0_decoder = None
    if is_discrete_f0:
        assert dataset.discrete_f0
        f0_decoder = Naive_F0_Decoder(
            args.f0_discretization_bounds, dataset.config.f0_vq_n_units
        ).cuda()

    if is_discrete_f0:
        assert f0_decoder  # correlation on tokens is meaningless

    dataset = InferenceDataset(
        dataset,
        args.prefix_length,
        filter_short=True,
        presort_by_length=True,
        min_length=args.min_length,
    )
    sampler = (
        None
        if world_size == 1
        else DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collater,
        sampler=sampler,
    )

    Ts = args.T_token, args.T_duration, args.T_f0
    decoder = TemperatureDecoder(
        Ts, discrete_dur=is_discrete_duration, discrete_f0=is_discrete_f0
    )

    mean_dur_prefix, mean_dur_cont = [], []
    mean_f0_prefix, mean_f0_cont = [], []

    for batch in dataloader:
        batch = explode_batch(batch, args.batch_explosion_rate)
        batch = move_to_cuda(batch)

        assert len(batch["prefix"]) == 1

        if args.teacher_force_tokens:
            autoregressive_steps = batch["target"].size(1) - args.prefix_length - 1
        else:
            autoregressive_steps = args.max_length - args.prefix_length  # + max_shift?

        if args.copy_target:
            produced_durations, produced_f0 = batch["dur_target"], batch["f0_target"]
        else:
            _, produced_durations, produced_f0, outputs = do_sampling(
                model,
                batch,
                tgt_dict.eos(),
                decoder,
                autoregressive_steps=autoregressive_steps,
                teacher_force_tokens=args.teacher_force_tokens,
                teacher_force_duration=args.teacher_force_duration,
                teacher_force_f0=args.teacher_force_f0,
            )

            # first tokens actually correspond to BOS
            produced_durations = produced_durations[:, 1:]
            produced_f0 = produced_f0[:, 1:]

        dur_target = batch["dur_target"]
        if is_discrete_duration:
            produced_durations = produced_durations.float()
            dur_target = dur_target.float()

        if is_discrete_f0:
            produced_f0 = f0_decoder(produced_f0).squeeze(-1)
            f0_target = batch["raw_f0"]
        else:
            f0_target = batch["f0_target"]

        # prefix values
        prefix = batch["prefix"][0]
        dur_prefix_mean = dur_target[:, :prefix].sum(dim=-1) / (
            (~batch["dur_mask"][:, :prefix]).sum(dim=-1)
        )

        non_voiced = f0_target[:, :prefix] == 0.0
        f0_mask = batch["f0_mask"][:, :prefix].logical_or(non_voiced)
        f0_prefix_mean = f0_target[:, :prefix].sum(dim=-1) / ((~f0_mask).sum(dim=-1))

        # continuation values
        dur_cont_mean = produced_durations[:, prefix:].sum(dim=-1) / (
            (~batch["dur_mask"][:, prefix:]).sum(dim=-1)
        )

        non_voiced = produced_f0[:, prefix:] == 0.0
        f0_mask = non_voiced
        f0_cont_mean = produced_f0[:, prefix:].sum(dim=-1) / ((~f0_mask).sum(dim=-1))

        assert not f0_cont_mean.isnan().any()

        mean_dur_prefix.append(dur_prefix_mean.cpu())
        mean_dur_cont.append(dur_cont_mean.cpu())

        mean_f0_prefix.append(f0_prefix_mean.cpu())
        mean_f0_cont.append(f0_cont_mean.cpu())

        if args.debug and len(mean_dur_prefix) > 10:
            break

    mean_dur_prefix, mean_dur_cont = torch.cat(mean_dur_prefix), torch.cat(
        mean_dur_cont
    )
    mean_f0_prefix, mean_f0_cont = torch.cat(mean_f0_prefix), torch.cat(mean_f0_cont)

    return mean_dur_prefix, mean_dur_cont, mean_f0_prefix, mean_f0_cont


def main(rank, world_size, args):
    start = time.time()

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="gloo", init_method="env://", world_size=world_size, rank=rank
        )
        torch.cuda.set_device(rank % torch.cuda.device_count())

    raw_args = args

    args = convert_namespace_to_omegaconf(args)
    if args.common.seed is not None:
        np.random.seed(args.common.seed)
        utils.set_torch_seed(args.common.seed)

    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [raw_args.path], arg_overrides={"data": args.task.data}
    )

    tgt_dict = task.target_dictionary

    for model in models:
        model.prepare_for_inference_(args)
        model.cuda().eval()
        if raw_args.fp16:
            model = model.half()
    model = models[0]

    config = ExpressiveCodeDataConfig(args.task.data)

    dataset = CodeDataset(
        manifest=config.manifests[raw_args.eval_subset],
        dictionary=task.source_dictionary,
        dur_dictionary=task.source_duration_dictionary,
        f0_dictionary=task.source_f0_dictionary,
        config=config,
        discrete_dur=task.cfg.discrete_duration,
        discrete_f0=task.cfg.discrete_f0,
        log_f0=task.cfg.log_f0,
        normalize_f0_mean=task.cfg.normalize_f0_mean,
        normalize_f0_std=task.cfg.normalize_f0_std,
        interpolate_f0=task.cfg.interpolate_f0,
        shifts=task.cfg.stream_shifts,
        return_filename=True,
        strip_filename=False,
        return_continuous_f0=raw_args.dequantize_prosody,
    )

    if raw_args.filter_names:
        dataset = FilterNamesDataset(dataset, raw_args.filter_names)

    criterion = task.build_criterion(model_args.criterion)

    name2metric = {
        "continuation": continuation,
        "teacher_force_everything": teacher_force_everything,
        "correlation": correlation,
    }

    name2keys = {
        "continuation": (
            "Token BLEU3",
            "Duration NLL",
            "Duration MAE",
            "F0 NLL",
            "F0 MAE",
            "F0 sum",
            "F0 sum_sq",
            "Dur sum",
            "Dur sum_sq",
        ),
        "teacher_force_everything": ("token_loss", "duration_loss", "f0_loss"),
        "correlation": ("Duration corr", "F0 corr"),
    }
    metric_name = raw_args.metric

    metric = name2metric[metric_name]
    results = metric(raw_args, dataset, model, criterion, tgt_dict, rank, world_size)

    values = None

    if metric_name not in [
        "correlation",
    ]:
        values, normalizers = results
        values = maybe_aggregate_normalize(values, normalizers, world_size)
    elif metric_name == "correlation":
        values = maybe_aggregate_correlations(results, world_size)
    else:
        assert False

    assert values is not None
    summary = dict(zip(name2keys[raw_args.metric], values.tolist()))
    if metric_name == "continuation":
        summary["F0 Std"] = np.sqrt(-summary["F0 sum"] ** 2 + summary["F0 sum_sq"])
        summary["Dur Std"] = np.sqrt(-summary["Dur sum"] ** 2 + summary["Dur sum_sq"])
        del summary["F0 sum"]
        del summary["F0 sum_sq"]
        del summary["Dur sum"]
        del summary["Dur sum_sq"]

    summary["metric"] = metric_name

    if rank == 0:
        print(summary)
        if raw_args.wandb:
            wandb_results(summary, raw_args)
        print("# finished in ", time.time() - start, "seconds")


def wandb_results(summary, raw_args):
    import wandb

    run = wandb.init(
        project=raw_args.wandb_project_name, tags=raw_args.wandb_tags.split(",")
    )
    run.config.metric = raw_args.metric
    run.config.model = raw_args.path
    run.config.data = raw_args.data

    if raw_args.wandb_run_name:
        run.name = raw_args.wandb_run_name
        run.save()

    wandb.log(summary)
    wandb.finish()


def maybe_aggregate_normalize(values, normalizers, world_size):
    if world_size > 1:
        torch.distributed.barrier()

        torch.distributed.all_reduce_multigpu([values])
        torch.distributed.all_reduce_multigpu([normalizers])

    return values / normalizers


def maybe_aggregate_correlations(results, world_size):
    if world_size > 1:
        output = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(output, results)
        mean_dur_prefix, mean_dur_cont, mean_f0_prefix, mean_f0_cont = [
            torch.cat([x[i] for x in output]) for i in range(4)
        ]
    else:
        mean_dur_prefix, mean_dur_cont, mean_f0_prefix, mean_f0_cont = results

    corr_dur = scipy.stats.pearsonr(mean_dur_prefix.numpy(), mean_dur_cont.numpy())[0]
    corr_f0 = scipy.stats.pearsonr(mean_f0_prefix.numpy(), mean_f0_cont.numpy())[0]
    values = torch.tensor([corr_dur, corr_f0])

    return values


def cli_main():
    parser = options.get_interactive_generation_parser()
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=1,
        help="Prompt prefix length (including <s>)",
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1,
        help="Multiply durations by the given scaler",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Process only the first batch"
    )
    parser.add_argument("--n_hypotheses", type=int, default=1)
    parser.add_argument("--filter-names", type=str, default=None)
    parser.add_argument(
        "--max-length", type=int, default=200, help="Maximal produced length"
    )

    parser.add_argument("--teacher-force-tokens", action="store_true", default=False)
    parser.add_argument("--teacher-force-duration", action="store_true", default=False)
    parser.add_argument("--teacher-force-f0", action="store_true", default=False)

    parser.add_argument("--copy-target", action="store_true", default=False)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--f0-discretization-bounds", type=str, default=None)
    parser.add_argument("--dequantize-prosody", action="store_true")
    parser.add_argument("--batch-explosion-rate", type=int, default=1)

    parser.add_argument(
        "--metric",
        choices=["continuation", "teacher_force_everything", "correlation"],
        required=True,
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project-name", type=str, default="eslm")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")

    parser.add_argument("--T-token", type=float, default=1.0)
    parser.add_argument("--T-duration", type=float, default=1.0)
    parser.add_argument("--T-f0", type=float, default=1.0)

    parser.add_argument("--n-workers", type=int, default=1)

    parser.add_argument(
        "--eval-subset", type=str, default="valid", choices=["valid", "test"]
    )

    args = options.parse_args_and_arch(parser)

    assert (
        args.prefix_length >= 1
    ), "Prefix length includes bos token <s>, hence the minimum is 1."
    assert args.temperature >= 0.0, "T must be non-negative!"

    if args.dequantize_prosody:
        assert args.f0_discretization_bounds

    world_size = args.n_workers or torch.cuda.device_count()
    if world_size > 1:
        import random

        mp.set_start_method("spawn", force=True)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(random.randint(10_000, 50_000))

        mp.spawn(
            main,
            nprocs=world_size,
            args=(
                world_size,
                args,
            ),
            join=True,
        )
    else:
        main(rank=0, world_size=world_size, args=args)


if __name__ == "__main__":
    cli_main()
