# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from . import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset

F0_FRAME_SPACE = 0.005  # sec


logger = logging.getLogger(__name__)


class ExpressiveCodeDataConfig(object):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.config = json.load(f)
        self._manifests = self.config["manifests"]

    @property
    def manifests(self):
        return self._manifests

    @property
    def n_units(self):
        return self.config["n_units"]

    @property
    def sampling_rate(self):
        return self.config["sampling_rate"]

    @property
    def code_hop_size(self):
        return self.config["code_hop_size"]

    @property
    def f0_stats(self):
        """pre-computed f0 statistics path"""
        return self.config.get("f0_stats", None)

    @property
    def f0_vq_type(self):
        """naive or precomp"""
        return self.config["f0_vq_type"]

    @property
    def f0_vq_name(self):
        return self.config["f0_vq_name"]

    def get_f0_vq_naive_quantizer(self, log, norm_mean, norm_std):
        key = "log" if log else "linear"
        if norm_mean and norm_std:
            key += "_mean_std_norm"
        elif norm_mean:
            key += "_mean_norm"
        else:
            key += "_none_norm"
        return self.config["f0_vq_naive_quantizer"][key]

    @property
    def f0_vq_n_units(self):
        return self.config["f0_vq_n_units"]

    @property
    def multispkr(self):
        """how to parse speaker label from audio path"""
        return self.config.get("multispkr", None)


def get_f0(audio, rate=16000):
    try:
        import amfm_decompy.basic_tools as basic
        import amfm_decompy.pYAAPT as pYAAPT
        from librosa.util import normalize
    except ImportError:
        raise "Please install amfm_decompy (`pip install AMFM-decompy`) and librosa (`pip install librosa`)."

    assert audio.ndim == 1
    frame_length = 20.0  # ms
    to_pad = int(frame_length / 1000 * rate) // 2

    audio = normalize(audio) * 0.95
    audio = np.pad(audio, (to_pad, to_pad), "constant", constant_values=0)
    audio = basic.SignalObj(audio, rate)
    pitch = pYAAPT.yaapt(
        audio,
        frame_length=frame_length,
        frame_space=F0_FRAME_SPACE * 1000,
        nccf_thresh1=0.25,
        tda_frame_length=25.0,
    )
    f0 = pitch.samp_values
    return f0


def interpolate_f0(f0):
    try:
        from scipy.interpolate import interp1d
    except ImportError:
        raise "Please install scipy (`pip install scipy`)"

    orig_t = np.arange(f0.shape[0])
    f0_interp = f0[:]
    ii = f0_interp != 0
    if ii.sum() > 1:
        f0_interp = interp1d(
            orig_t[ii], f0_interp[ii], bounds_error=False, kind="linear", fill_value=0
        )(orig_t)
        f0_interp = torch.Tensor(f0_interp).type_as(f0).to(f0.device)
    return f0_interp


def naive_quantize(x, edges):
    bin_idx = (x.view(-1, 1) > edges.view(1, -1)).long().sum(dim=1)
    return bin_idx


def load_wav(full_path):
    try:
        import soundfile as sf
    except ImportError:
        raise "Please install soundfile (`pip install SoundFile`)"
    data, sampling_rate = sf.read(full_path)
    return data, sampling_rate


def parse_code(code_str, dictionary, append_eos):
    code, duration = torch.unique_consecutive(
        torch.ShortTensor(list(map(int, code_str.split()))), return_counts=True
    )
    code = " ".join(map(str, code.tolist()))
    code = dictionary.encode_line(code, append_eos).short()

    if append_eos:
        duration = torch.cat((duration, duration.new_zeros((1,))), dim=0)  # eos
    duration = duration.short()
    return code, duration


def parse_manifest(manifest, dictionary):
    audio_files = []
    codes = []
    durations = []
    speakers = []

    with open(manifest) as info:
        for line in info.readlines():
            sample = eval(line.strip())
            if "cpc_km100" in sample:
                k = "cpc_km100"
            elif "hubert_km100" in sample:
                k = "hubert_km100"
            elif "phone" in sample:
                k = "phone"
            else:
                assert False, "unknown format"
            code = sample[k]
            code, duration = parse_code(code, dictionary, append_eos=True)

            codes.append(code)
            durations.append(duration)
            audio_files.append(sample["audio"])
            speakers.append(sample.get("speaker", None))

    return audio_files, codes, durations, speakers


def parse_speaker(path, method):
    if type(path) == str:
        path = Path(path)

    if method == "parent_name":
        return path.parent.name
    elif method == "parent_parent_name":
        return path.parent.parent.name
    elif method == "_":
        return path.name.split("_")[0]
    elif method == "single":
        return "A"
    elif callable(method):
        return method(path)
    else:
        raise NotImplementedError()


def get_f0_by_filename(filename, tgt_sampling_rate):
    audio, sampling_rate = load_wav(filename)
    if sampling_rate != tgt_sampling_rate:
        raise ValueError(
            "{} SR doesn't match target {} SR".format(sampling_rate, tgt_sampling_rate)
        )

    # compute un-interpolated f0, and use Ann's interp in __getitem__ if set
    f0 = get_f0(audio, rate=tgt_sampling_rate)
    f0 = torch.from_numpy(f0.astype(np.float32))
    return f0


def align_f0_to_durations(f0, durations, f0_code_ratio, tol=1):
    code_len = durations.sum()
    targ_len = int(f0_code_ratio * code_len)
    diff = f0.size(0) - targ_len
    assert abs(diff) <= tol, (
        f"Cannot subsample F0: |{f0.size(0)} - {f0_code_ratio}*{code_len}|"
        f" > {tol} (dur=\n{durations})"
    )
    if diff > 0:
        f0 = f0[:targ_len]
    elif diff < 0:
        f0 = torch.cat((f0, f0.new_full((-diff,), f0[-1])), 0)

    f0_offset = 0.0
    seg_f0s = []
    for dur in durations:
        f0_dur = dur.item() * f0_code_ratio
        seg_f0 = f0[int(f0_offset) : int(f0_offset + f0_dur)]
        seg_f0 = seg_f0[seg_f0 != 0]
        if len(seg_f0) == 0:
            seg_f0 = torch.tensor(0).type(seg_f0.type())
        else:
            seg_f0 = seg_f0.mean()
        seg_f0s.append(seg_f0)
        f0_offset += f0_dur

    assert int(f0_offset) == f0.size(0), f"{f0_offset} {f0.size()} {durations.sum()}"
    return torch.tensor(seg_f0s)


class Paddings(object):
    def __init__(self, code_val, dur_val=0, f0_val=-2.0):
        self.code = code_val
        self.dur = dur_val
        self.f0 = f0_val


class Shifts(object):
    def __init__(self, shifts_str, pads):
        self._shifts = list(map(int, shifts_str.split(",")))
        assert len(self._shifts) == 2, self._shifts
        assert all(s >= 0 for s in self._shifts)
        self.extra_length = max(s for s in self._shifts)
        self.pads = pads

    @property
    def dur(self):
        return self._shifts[0]

    @property
    def f0(self):
        return self._shifts[1]

    @staticmethod
    def shift_one(seq, left_pad_num, right_pad_num, pad):
        assert seq.ndim == 1
        bos = seq.new_full((left_pad_num,), pad)
        eos = seq.new_full((right_pad_num,), pad)
        seq = torch.cat([bos, seq, eos])
        mask = torch.ones_like(seq).bool()
        mask[left_pad_num : len(seq) - right_pad_num] = 0
        return seq, mask

    def __call__(self, code, dur, f0):
        if self.extra_length == 0:
            code_mask = torch.zeros_like(code).bool()
            dur_mask = torch.zeros_like(dur).bool()
            f0_mask = torch.zeros_like(f0).bool()
            return code, code_mask, dur, dur_mask, f0, f0_mask

        code, code_mask = self.shift_one(code, 0, self.extra_length, self.pads.code)
        dur, dur_mask = self.shift_one(
            dur, self.dur, self.extra_length - self.dur, self.pads.dur
        )
        f0, f0_mask = self.shift_one(
            f0, self.f0, self.extra_length - self.f0, self.pads.f0
        )
        return code, code_mask, dur, dur_mask, f0, f0_mask


class CodeDataset(FairseqDataset):
    def __init__(
        self,
        manifest,
        dictionary,
        dur_dictionary,
        f0_dictionary,
        config,
        discrete_dur,
        discrete_f0,
        log_f0,
        normalize_f0_mean,
        normalize_f0_std,
        interpolate_f0,
        return_filename=False,
        strip_filename=True,
        shifts="0,0",
        return_continuous_f0=False,
    ):
        random.seed(1234)
        self.dictionary = dictionary
        self.dur_dictionary = dur_dictionary
        self.f0_dictionary = f0_dictionary
        self.config = config

        # duration config
        self.discrete_dur = discrete_dur

        # pitch config
        self.discrete_f0 = discrete_f0
        self.log_f0 = log_f0
        self.normalize_f0_mean = normalize_f0_mean
        self.normalize_f0_std = normalize_f0_std
        self.interpolate_f0 = interpolate_f0

        self.return_filename = return_filename
        self.strip_filename = strip_filename
        self.f0_code_ratio = config.code_hop_size / (
            config.sampling_rate * F0_FRAME_SPACE
        )

        # use lazy loading to avoid sharing file handlers across workers
        self.manifest = manifest
        self._codes = None
        self._durs = None
        self._f0s = None
        with open(f"{manifest}.leng.txt", "r") as f:
            lengs = [int(line.rstrip()) for line in f]
            edges = np.cumsum([0] + lengs)
            self.starts, self.ends = edges[:-1], edges[1:]
        with open(f"{manifest}.path.txt", "r") as f:
            self.file_names = [line.rstrip() for line in f]
        logger.info(f"num entries: {len(self.starts)}")

        if os.path.exists(f"{manifest}.f0_stat.pt"):
            self.f0_stats = torch.load(f"{manifest}.f0_stat.pt")
        elif config.f0_stats:
            self.f0_stats = torch.load(config.f0_stats)

        self.multispkr = config.multispkr
        if config.multispkr:
            with open(f"{manifest}.speaker.txt", "r") as f:
                self.spkrs = [line.rstrip() for line in f]
            self.id_to_spkr = sorted(self.spkrs)
            self.spkr_to_id = {k: v for v, k in enumerate(self.id_to_spkr)}

        self.pads = Paddings(
            dictionary.pad(),
            0,  # use 0 for duration padding
            f0_dictionary.pad() if discrete_f0 else -5.0,
        )
        self.shifts = Shifts(shifts, pads=self.pads)
        self.return_continuous_f0 = return_continuous_f0

    def get_data_handlers(self):
        logging.info(f"loading data for {self.manifest}")
        self._codes = np.load(f"{self.manifest}.code.npy", mmap_mode="r")
        self._durs = np.load(f"{self.manifest}.dur.npy", mmap_mode="r")

        if self.discrete_f0:
            if self.config.f0_vq_type == "precomp":
                self._f0s = np.load(
                    f"{self.manifest}.{self.config.f0_vq_name}.npy", mmap_mode="r"
                )
            elif self.config.f0_vq_type == "naive":
                self._f0s = np.load(f"{self.manifest}.f0.npy", mmap_mode="r")
                quantizers_path = self.config.get_f0_vq_naive_quantizer(
                    self.log_f0, self.normalize_f0_mean, self.normalize_f0_std
                )
                quantizers = torch.load(quantizers_path)
                n_units = self.config.f0_vq_n_units
                self._f0_quantizer = torch.from_numpy(quantizers[n_units])
            else:
                raise ValueError(f"f0_vq_type {self.config.f0_vq_type} not supported")
        else:
            self._f0s = np.load(f"{self.manifest}.f0.npy", mmap_mode="r")

    def preprocess_f0(self, f0, stats):
        """
        1. interpolate
        2. log transform (keep unvoiced frame 0)
        """
        # TODO: change this to be dependent on config for naive quantizer
        f0 = f0.clone()
        if self.interpolate_f0:
            f0 = interpolate_f0(f0)

        mask = f0 != 0  # only process voiced frames
        if self.log_f0:
            f0[mask] = f0[mask].log()
        if self.normalize_f0_mean:
            mean = stats["logf0_mean"] if self.log_f0 else stats["f0_mean"]
            f0[mask] = f0[mask] - mean
        if self.normalize_f0_std:
            std = stats["logf0_std"] if self.log_f0 else stats["f0_std"]
            f0[mask] = f0[mask] / std
        return f0

    def _get_raw_item(self, index):
        start, end = self.starts[index], self.ends[index]
        if self._codes is None:
            self.get_data_handlers()
        code = torch.from_numpy(np.array(self._codes[start:end])).long()
        dur = torch.from_numpy(np.array(self._durs[start:end]))
        f0 = torch.from_numpy(np.array(self._f0s[start:end]))
        return code, dur, f0

    def __getitem__(self, index):
        code, dur, f0 = self._get_raw_item(index)
        code = torch.cat([code.new([self.dictionary.bos()]), code])

        # use 0 for eos and bos
        dur = torch.cat([dur.new([0]), dur])
        if self.discrete_dur:
            dur = self.dur_dictionary.encode_line(
                " ".join(map(str, dur.tolist())), append_eos=False
            ).long()
        else:
            dur = dur.float()

        # TODO: find a more elegant approach
        raw_f0 = None
        if self.discrete_f0:
            if self.config.f0_vq_type == "precomp":
                f0 = self.f0_dictionary.encode_line(
                    " ".join(map(str, f0.tolist())), append_eos=False
                ).long()
            else:
                f0 = f0.float()
                f0 = self.preprocess_f0(f0, self.f0_stats[self.spkrs[index]])
                if self.return_continuous_f0:
                    raw_f0 = f0
                    raw_f0 = torch.cat([raw_f0.new([self.f0_dictionary.bos()]), raw_f0])
                f0 = naive_quantize(f0, self._f0_quantizer)
            f0 = torch.cat([f0.new([self.f0_dictionary.bos()]), f0])
        else:
            f0 = f0.float()
            if self.multispkr:
                f0 = self.preprocess_f0(f0, self.f0_stats[self.spkrs[index]])
            else:
                f0 = self.preprocess_f0(f0, self.f0_stats)
            f0 = torch.cat([f0.new([0]), f0])

        if raw_f0 is not None:
            *_, raw_f0, raw_f0_mask = self.shifts(code, dur, raw_f0)
        else:
            raw_f0_mask = None

        code, code_mask, dur, dur_mask, f0, f0_mask = self.shifts(code, dur, f0)
        if raw_f0_mask is not None:
            assert (raw_f0_mask == f0_mask).all()

        # is a padded frame if either input or output is padded
        feats = {
            "source": code[:-1],
            "target": code[1:],
            "mask": code_mask[1:].logical_or(code_mask[:-1]),
            "dur_source": dur[:-1],
            "dur_target": dur[1:],
            "dur_mask": dur_mask[1:].logical_or(dur_mask[:-1]),
            "f0_source": f0[:-1],
            "f0_target": f0[1:],
            "f0_mask": f0_mask[1:].logical_or(f0_mask[:-1]),
        }

        if raw_f0 is not None:
            feats["raw_f0"] = raw_f0[1:]

        if self.return_filename:
            fname = self.file_names[index]
            feats["filename"] = (
                fname if not self.strip_filename else Path(fname).with_suffix("").name
            )
        return feats

    def __len__(self):
        return len(self.starts)

    def size(self, index):
        return self.ends[index] - self.starts[index] + self.shifts.extra_length

    def num_tokens(self, index):
        return self.size(index)

    def collater(self, samples):
        pad_idx, eos_idx = self.dictionary.pad(), self.dictionary.eos()
        if len(samples) == 0:
            return {}

        src_tokens = data_utils.collate_tokens(
            [s["source"] for s in samples], pad_idx, eos_idx, left_pad=False
        )

        tgt_tokens = data_utils.collate_tokens(
            [s["target"] for s in samples],
            pad_idx=pad_idx,
            eos_idx=pad_idx,  # appending padding, eos is there already
            left_pad=False,
        )

        src_durs, tgt_durs = [
            data_utils.collate_tokens(
                [s[k] for s in samples],
                pad_idx=self.pads.dur,
                eos_idx=self.pads.dur,
                left_pad=False,
            )
            for k in ["dur_source", "dur_target"]
        ]

        src_f0s, tgt_f0s = [
            data_utils.collate_tokens(
                [s[k] for s in samples],
                pad_idx=self.pads.f0,
                eos_idx=self.pads.f0,
                left_pad=False,
            )
            for k in ["f0_source", "f0_target"]
        ]

        mask, dur_mask, f0_mask = [
            data_utils.collate_tokens(
                [s[k] for s in samples],
                pad_idx=1,
                eos_idx=1,
                left_pad=False,
            )
            for k in ["mask", "dur_mask", "f0_mask"]
        ]

        src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
        n_tokens = sum(len(s["source"]) for s in samples)

        result = {
            "nsentences": len(samples),
            "ntokens": n_tokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "dur_src": src_durs,
                "f0_src": src_f0s,
            },
            "target": tgt_tokens,
            "dur_target": tgt_durs,
            "f0_target": tgt_f0s,
            "mask": mask,
            "dur_mask": dur_mask,
            "f0_mask": f0_mask,
        }

        if "filename" in samples[0]:
            result["filename"] = [s["filename"] for s in samples]

        # TODO: remove this hack into the inference dataset
        if "prefix" in samples[0]:
            result["prefix"] = [s["prefix"] for s in samples]

        if "raw_f0" in samples[0]:
            raw_f0s = data_utils.collate_tokens(
                [s["raw_f0"] for s in samples],
                pad_idx=self.pads.f0,
                eos_idx=self.pads.f0,
                left_pad=False,
            )
            result["raw_f0"] = raw_f0s
        return result
