# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch.multiprocessing as mp
import numpy as np
import json

import torch
from torch.distributions.categorical import Categorical

from fairseq import checkpoint_utils, options, utils
from fairseq.data.codedataset import CodeDataset, ExpressiveCodeDataConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from torch.utils.data import DataLoader, DistributedSampler
from fairseq.utils import move_to_cuda

import tqdm
import random
import pathlib

import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from inference_dataset import InferenceDataset, explode_batch
from naive_decoder import Naive_F0_Decoder
from truncated_laplace import truncated_laplace

CODETYPE_TO_FRAMETIME = {"cpc_km100": 0.01, "hubert": 0.02}  # 10ms  # 20ms


class TemperatureDecoder:
    def __init__(self, Ts, discrete_dur=False, discrete_f0=False):
        self.T_token, self.T_dur, self.T_f0 = Ts
        self.discrete_dur = discrete_dur
        self.discrete_f0 = discrete_f0

    def __call__(self, output):
        def sample_multinomial(key, T):
            logits = output[key][:, -1, :].float()
            return Categorical(logits=logits / T).sample().unsqueeze(-1)

        def sample_laplace(key, T, truncate_at_zero):
            mean = output[key][:, -1, :].float()
            return truncated_laplace(mean=mean, T=T, truncate_by_zero=truncate_at_zero)

        if self.T_token > 0:
            new_tokens = sample_multinomial("token", self.T_token)
        else:
            new_tokens = output["token"][:, -1, :].argmax(dim=-1, keepdim=True)

        if not self.discrete_dur and self.T_dur == 0:
            new_durations = output["duration"][:, -1].round().int()
        elif not self.discrete_dur and self.T_dur > 0:
            new_durations = (
                sample_laplace("duration", self.T_dur, truncate_at_zero=True)
                .round()
                .int()
            )
        elif self.discrete_dur and self.T_dur > 0:
            new_durations = sample_multinomial("duration", self.T_dur)
        elif self.discrete_dur and self.T_dur == 0:
            new_durations = output["duration"][:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            assert False

        if not self.discrete_f0 and self.T_f0 == 0:
            new_f0 = output["f0"][:, -1]
        elif not self.discrete_f0 and self.T_f0 > 0:
            new_f0 = sample_laplace("f0", self.T_f0, truncate_at_zero=False)
        elif self.discrete_f0 and self.T_f0 > 0:
            new_f0 = sample_multinomial("f0", self.T_f0)
        elif self.discrete_f0 and self.T_f0 == 0:
            new_f0 = output["f0"][:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            assert False

        return new_tokens, new_durations, new_f0


class FilterNamesDataset:
    def __init__(self, dataset, fnames_path):
        self.dataset = dataset

        with open(fnames_path, "r") as fin:
            fnames = set((eval(line)["audio"] for line in fin))
            print(f"# will retrict the dataset for {len(fnames)} files")

        self.indexes = []

        for i, datapoint in enumerate(dataset):
            if datapoint["filename"] in fnames:
                self.indexes.append(i)
        assert len(self.indexes) == len(fnames), f"{len(self.indexes)} {len(fnames)}"

        self.collater = self.dataset.collater
        self.discrete_dur = self.dataset.discrete_dur
        self.discrete_f0 = self.dataset.discrete_f0

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, k):
        k = self.indexes[k]
        return self.dataset[k]

    def size(self, k):
        k = self.indexes[k]
        return self.dataset.size(k)


@torch.no_grad()
def do_sampling(
    model,
    batch,
    eos_token,
    decoder,
    autoregressive_steps=100,
    teacher_force_tokens=False,
    teacher_force_duration=False,
    teacher_force_f0=False,
    match_duration=False,
):
    def autoregressive_step_(output, autoregressive_steps):
        new_tokens, new_durations, new_f0 = decoder(output)

        n = output["token"].size(1) if output["token"].ndim == 3 else 1

        if teacher_force_tokens:
            new_tokens = batch["target"][:, n - 1].unsqueeze(-1)
        if teacher_force_duration:
            new_durations = batch["dur_target"][:, n - 1].unsqueeze(-1)
        if teacher_force_f0:
            new_f0 = batch["f0_target"][:, n - 1].unsqueeze(-1)

        batch["net_input"]["src_tokens"] = torch.cat(
            [batch["net_input"]["src_tokens"], new_tokens], dim=1
        )
        batch["net_input"]["dur_src"] = torch.cat(
            [batch["net_input"]["dur_src"], new_durations], dim=1
        )
        batch["net_input"]["f0_src"] = torch.cat(
            [batch["net_input"]["f0_src"], new_f0], dim=1
        )

    outputs = []

    if teacher_force_tokens or teacher_force_duration or teacher_force_f0:
        max_time = batch["target"].size(1)
        prefix_time = batch["net_input"]["src_tokens"].size(1)

        autoregressive_steps = max_time - prefix_time + 1  # should be 0

    for _ in range(autoregressive_steps):
        output = model(**batch["net_input"])

        last_steps = (
            output["token"][:, -1, ...],
            output["duration"][:, -1, ...],
            output["f0"][:, -1, ...],
        )
        outputs.append(last_steps)

        autoregressive_step_(output, autoregressive_steps)
        tokens, duration, f0 = (
            batch["net_input"]["src_tokens"],
            batch["net_input"]["dur_src"],
            batch["net_input"]["f0_src"],
        )

        if (
            match_duration
            and (batch["dur_target"].sum(dim=-1) < duration.sum(dim=-1)).all()
        ):
            break

    return tokens, duration, f0, outputs


def unroll_duration(token_stream, duration_stream):
    assert len(token_stream) == len(
        duration_stream
    ), f"{len(token_stream)} != {len(duration_stream)}"
    non_positive_durations = sum(d <= 0 for d in duration_stream)
    if non_positive_durations > 0:
        print(
            f"# {non_positive_durations} durations are non-positive, they will be capped to 1"
        )

    result = []

    duration_stream_rounded_capped = [max(1, int(round(x))) for x in duration_stream]
    for t, d in zip(token_stream, duration_stream_rounded_capped):
        result.extend([t] * d)

    return result


def realign_shifted_streams(tokens, durations, F0s, shifts):
    """
    Durations are shifted by 1, F0 by 2
    >>> tokens = ["<s>", "t1",  "t2", "t3", "</s>", "x", "x"]
    >>> durations = ["<0>", "<0>", "d1", "d2", "d3", "<0>", "x"]
    >>> F0s    = ["<0>", "<0>", "<0>", "f1", "f2", "f3", "<0>"]
    >>> shifts = [1,2]
    >>> realign_shifted_streams(tokens, durations, F0s, shifts)
    (['<s>', 't1', 't2', 't3', '</s>'], ['<0>', 'd1', 'd2', 'd3', '<0>'], ['<0>', 'f1', 'f2', 'f3', '<0>'])
    """
    max_shift = max(shifts)
    if max_shift > 0:
        shift_durations, shift_F0s = shifts

        tokens = tokens[:-max_shift]
        durations = durations[shift_durations:]
        if shift_durations < max_shift:
            durations = durations[: -(max_shift - shift_durations)]

        if F0s is not None:
            F0s = F0s[shift_F0s:]
            if shift_F0s < max_shift:
                F0s = F0s[: -(max_shift - shift_F0s)]

    assert len(tokens) == len(durations), f"{len(tokens)} =! {len(durations)}"
    if F0s is not None:
        assert len(tokens) == len(F0s), f"{len(tokens)} =! {len(F0s)}"

    return tokens, durations, F0s


def maybe_cut_eos(produced_tokens, produced_duration, produced_f0, eos_idx):
    if eos_idx in produced_tokens:
        eos_index = produced_tokens.index(eos_idx)
        produced_tokens = produced_tokens[:eos_index]
        produced_duration = produced_duration[:eos_index]
        produced_f0 = produced_f0[:eos_index]
    return produced_tokens, produced_duration, produced_f0


def maybe_filter_pad(produced_tokens, produced_duration, produced_f0, pad_idx):
    if pad_idx not in produced_tokens:
        return produced_tokens, produced_duration, produced_f0

    assert len(produced_tokens) == len(produced_duration) == len(produced_f0)

    print("<pad> is detected in the output!")
    filtered_tokens, filtered_duration, filtered_f0 = [], [], []

    for t, d, f in zip(produced_tokens, produced_duration, produced_f0):
        if t != pad_idx:
            filtered_tokens.append(t)
            filtered_duration.append(d)
            filtered_f0.append(f)
    return filtered_tokens, filtered_duration, filtered_f0


def match_duration(produced_tokens, produced_duration, produced_f0, target_duration):
    """
    >>> tokens = ['t'] * 4
    >>> F0s    = ['f0'] * 4
    >>> produced_duration = [1, 10, 10, 10]
    >>> match_duration(tokens, produced_duration, F0s, target_duration=100)
    (['t', 't', 't', 't'], [1, 10, 10, 10], ['f0', 'f0', 'f0', 'f0'])
    >>> match_duration(tokens, produced_duration, F0s, target_duration=5)
    (['t', 't'], [1, 4], ['f0', 'f0'])
    """
    if sum(produced_duration) <= target_duration:
        return produced_tokens, produced_duration, produced_f0

    running_duration = 0
    filtered_duration = []

    for next_tok_duration in produced_duration:
        if running_duration + next_tok_duration < target_duration:
            filtered_duration.append(next_tok_duration)
            running_duration += next_tok_duration
        else:
            to_add = target_duration - running_duration
            assert to_add <= next_tok_duration
            filtered_duration.append(to_add)
            break

    produced_duration = filtered_duration
    assert sum(produced_duration) == target_duration

    n_tok = len(filtered_duration)

    return produced_tokens[:n_tok], produced_duration, produced_f0[:n_tok]


def main(rank, world_size, args):
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="gloo", init_method="env://", world_size=world_size, rank=rank
        )
        torch.cuda.set_device(rank)

    raw_args = args
    args = convert_namespace_to_omegaconf(args)
    if args.common.seed is not None:
        random.seed(args.common.seed)
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
        manifest=config.manifests[raw_args.subset],
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
    )
    tgt_dict = task.target_dictionary
    shifts = dataset.shifts.dur, dataset.shifts.f0
    max_shift = max(shifts)

    fname = raw_args.output
    if world_size > 1:
        fname += f"_{rank}"
    output_file = open(fname, "w")

    if raw_args.filter_names:
        dataset = FilterNamesDataset(dataset, raw_args.filter_names)

    dataset = InferenceDataset(dataset, raw_args.prefix_length, filter_short=True)
    print(f"Dataset size {len(dataset)}")
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

    Ts = raw_args.T_token, raw_args.T_duration, raw_args.T_f0
    decoder = TemperatureDecoder(
        Ts, discrete_dur=task.cfg.discrete_duration, discrete_f0=task.cfg.discrete_f0
    )

    dataset_size = len(dataset)

    f0_decoder = None
    if raw_args.f0_discretization_bounds:
        assert task.cfg.discrete_f0
        f0_decoder = Naive_F0_Decoder(raw_args.f0_discretization_bounds).cuda()

    pbar = (
        tqdm.tqdm(
            total=dataset_size
            if raw_args.max_samples is None
            else min(raw_args.max_samples, dataset_size)
        )
        if world_size == 1
        else None
    )

    samples_produced = 0

    for batch in dataloader:
        if (
            raw_args.max_samples is not None
            and samples_produced >= raw_args.max_samples
        ):
            break

        prefix = batch["prefix"][0]

        batch = explode_batch(batch, raw_args.batch_explosion_rate)
        batch = move_to_cuda(batch)

        if not raw_args.short_curcuit:
            produced_tokens, produced_durations, produced_f0, _ = do_sampling(
                models[0],
                batch,
                tgt_dict.eos(),
                decoder,
                autoregressive_steps=raw_args.max_length - prefix + max_shift,
                teacher_force_tokens=raw_args.teacher_force_tokens,
                match_duration=raw_args.match_duration,
                teacher_force_duration=raw_args.teacher_force_duration,
                teacher_force_f0=raw_args.teacher_force_f0,
            )

            # stip entries corresponding to <s>
            produced_tokens = produced_tokens[:, 1:]
            produced_durations = produced_durations[:, 1:]
            produced_f0 = produced_f0[:, 1:]

        else:
            max_length = raw_args.max_length + max_shift
            produced_tokens, produced_durations, produced_f0 = (
                batch["target"][:, :max_length],
                batch["dur_target"][:, :max_length],
                batch["f0_target"][:, :max_length],
            )

        if f0_decoder is not None:
            produced_f0 = f0_decoder(produced_f0)

        produced_tokens, produced_durations, produced_f0 = (
            produced_tokens.cpu().tolist(),
            produced_durations.cpu().tolist(),
            produced_f0.cpu().tolist(),
        )

        bsz = batch["target"].size(0)
        assert bsz == raw_args.batch_explosion_rate

        for i in range(bsz):
            if (
                raw_args.max_samples is not None
                and samples_produced >= raw_args.max_samples
            ):
                break

            produced_tokens_i = produced_tokens[i]
            produced_durations_i = produced_durations[i]
            produced_f0_i = produced_f0[i]

            (
                produced_tokens_i,
                produced_durations_i,
                produced_f0_i,
            ) = realign_shifted_streams(
                produced_tokens_i, produced_durations_i, produced_f0_i, shifts
            )

            produced_tokens_i, produced_durations_i, produced_f0_i = maybe_cut_eos(
                produced_tokens_i, produced_durations_i, produced_f0_i, tgt_dict.eos()
            )

            produced_tokens_i, produced_durations_i, produced_f0_i = maybe_filter_pad(
                produced_tokens_i, produced_durations_i, produced_f0_i, tgt_dict.pad()
            )

            if raw_args.match_duration:
                # NB: here we cheat a bit and use that padding has duration 0
                # so no need to re-align and remove padding
                dur_target_i = batch["dur_target"][i, :].sum().item()
                produced_tokens_i, produced_durations_i, produced_f0_i = match_duration(
                    produced_tokens_i, produced_durations_i, produced_f0_i, dur_target_i
                )

            if raw_args.cut_prompt:
                produced_tokens_i, produced_durations_i, produced_f0_i = (
                    produced_tokens_i[prefix:],
                    produced_durations_i[prefix:],
                    produced_f0_i[prefix:],
                )

            prompt_fname = batch["filename"][0]
            fname = str(pathlib.Path(prompt_fname).with_suffix("")) + f"__{i}.wav"

            token_stream = unroll_duration(produced_tokens_i, produced_durations_i)
            f0_stream = unroll_duration(produced_f0_i, produced_durations_i)
            output_line = json.dumps(
                {
                    "audio": fname,
                    "prompt": prompt_fname,
                    raw_args.code_type: " ".join(map(str, token_stream)),
                    "duration": round(
                        sum(produced_durations_i)
                        * CODETYPE_TO_FRAMETIME[raw_args.code_type],
                        3,
                    ),
                    "raw_duration": produced_durations_i,
                    "raw_f0": produced_f0_i,
                    "f0": [round(f0, 3) for f0 in f0_stream],
                }
            )
            print(output_line, file=output_file)

            if pbar:
                pbar.update(1)
            samples_produced += 1

        if raw_args.debug:
            break

    output_file.close()

    if world_size > 1:
        # important that everything is flushed before aggregating
        torch.distributed.barrier()

    if world_size > 1 and rank == 0:
        with open(raw_args.output, "w") as fout:
            for i in range(world_size):
                f = raw_args.output + f"_{i}"
                with open(f, "r") as fin:
                    fout.write(fin.read())
                os.remove(f)


def cli_main():
    parser = options.get_interactive_generation_parser()
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=1,
        help="Prompt prefix length (including <s>)",
    )
    parser.add_argument("--output", type=str, default=None, required=True)
    parser.add_argument(
        "--debug", action="store_true", help="Process only the first batch"
    )
    parser.add_argument(
        "--ignore-durations",
        action="store_true",
        help="If set, the duration stream is ignored",
    )
    parser.add_argument(
        "--max-length", type=int, default=200, help="Maximal produced length"
    )
    parser.add_argument(
        "--code-type", choices=["cpc_km100", "hubert"], default="cpc_km100"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prompt-duration-scaler", type=float, default=1.0)
    parser.add_argument("--teacher-force-tokens", action="store_true", default=False)
    parser.add_argument("--teacher-force-duration", action="store_true", default=False)
    parser.add_argument("--teacher-force-f0", action="store_true", default=False)
    parser.add_argument("--filter-names", type=str, default=None)
    parser.add_argument(
        "--match-duration",
        action="store_true",
        help="Do not produce sequences longer that ground-truth",
    )
    parser.add_argument(
        "--cut-prompt",
        action="store_true",
        help="Remove prompt from the produced audio",
    )
    parser.add_argument(
        "--short-curcuit", action="store_true", help="Use 'target' as a sample"
    )
    parser.add_argument("--f0-discretization-bounds", type=str, default=None)

    parser.add_argument("--batch-explosion-rate", type=int, default=1)

    parser.add_argument("--T-token", type=float, default=1.0)
    parser.add_argument("--T-duration", type=float, default=1.0)
    parser.add_argument("--T-f0", type=float, default=1.0)

    parser.add_argument(
        "--subset", type=str, default="valid", choices=["test", "valid"]
    )

    args = options.parse_args_and_arch(parser)

    assert (
        args.prefix_length >= 1
    ), "Prefix length includes bos token <s>, hence the minimum is 1."
    assert all(
        t >= 0 for t in [args.T_token, args.T_f0, args.T_duration]
    ), "T must be non-negative!"

    world_size = torch.cuda.device_count()
    if world_size > 1:
        import random

        mp.set_start_method("spawn", force=True)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(random.randint(10_000, 50_000))

        print(f"Using {world_size} devices, master port {os.environ['MASTER_PORT']}")

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
