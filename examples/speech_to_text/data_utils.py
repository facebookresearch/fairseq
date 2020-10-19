#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import os.path as op
import zipfile
from functools import reduce
from glob import glob
from multiprocessing import cpu_count
from typing import Any, Dict, List

import numpy as np
import sentencepiece as sp
from fairseq.data.audio.audio_utils import _get_kaldi_fbank, _get_torchaudio_fbank
from fairseq.data.audio.feature_transforms.utterance_cmvn import UtteranceCMVN
from tqdm import tqdm


UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1


def gen_vocab(
    input_path: str,
    output_path_prefix: str,
    model_type="bpe",
    vocab_size=1000,
):
    # Train SentencePiece Model
    arguments = [
        f"--input={input_path}",
        f"--model_prefix={output_path_prefix}",
        f"--model_type={model_type}",
        f"--vocab_size={vocab_size}",
        "--character_coverage=1.0",
        f"--num_threads={cpu_count()}",
        f"--unk_id={UNK_TOKEN_ID}",
        f"--bos_id={BOS_TOKEN_ID}",
        f"--eos_id={EOS_TOKEN_ID}",
        f"--pad_id={PAD_TOKEN_ID}",
    ]
    sp.SentencePieceTrainer.Train(" ".join(arguments))
    # Export fairseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix + ".model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (
        vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
        and vocab.get(PAD_TOKEN_ID) == PAD_TOKEN
        and vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
        and vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    )
    vocab = {
        i: s
        for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_path_prefix + ".txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")


def extract_fbank_features(
    waveform,
    sample_rate,
    output_path=None,
    n_mel_bins=80,
    apply_utterance_cmvn=True,
    overwrite=False,
):
    if output_path is not None and op.exists(output_path) and not overwrite:
        return

    _waveform = waveform * (2 ** 15)  # Kaldi compliance: 16-bit signed integers
    _waveform = _waveform.squeeze().numpy()

    features = _get_kaldi_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )

    if apply_utterance_cmvn:
        cmvn = UtteranceCMVN(norm_means=True, norm_vars=True)
        features = cmvn(features)
    if output_path is not None:
        np.save(output_path, features)
    else:
        return features


def create_zip(data_root, zip_path):
    cwd = os.path.abspath(os.curdir)
    os.chdir(data_root)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for filename in tqdm(glob("*.npy")):
            f.write(filename)
    os.chdir(cwd)


def is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78


def get_zip_manifest(zip_root, zip_filename):
    zip_path = op.join(zip_root, zip_filename)
    with zipfile.ZipFile(zip_path, mode="r") as f:
        info = f.infolist()
    manifest = {}
    for i in tqdm(info):
        utt_id = op.splitext(i.filename)[0]
        offset, file_size = i.header_offset + 30 + len(i.filename), i.file_size
        manifest[utt_id] = f"{zip_filename}:{offset}:{file_size}"
        with open(zip_path, "rb") as f:
            f.seek(offset)
            data = f.read(file_size)
            assert len(data) > 1 and is_npy_data(data)
    return manifest


def gen_config_yaml(
    data_root, spm_filename, yaml_filename="config.yaml", specaugment_policy="lb"
):
    assert specaugment_policy in {"lb", "ld"}
    data_root = op.abspath(data_root)
    writer = S2TDataConfigWriter(op.join(data_root, yaml_filename))
    writer.set_audio_root(op.abspath(data_root))
    writer.set_vocab_filename(spm_filename.replace(".model", ".txt"))
    writer.set_input_channels(1)
    writer.set_input_feat_per_channel(80)
    if specaugment_policy == "lb":
        writer.set_specaugment_lb_policy()
    else:
        writer.set_specaugment_ld_policy()
    writer.set_bpe_tokenizer(
        {
            "bpe": "sentencepiece",
            "sentencepiece_model": op.join(data_root, spm_filename),
        }
    )
    writer.set_feature_transforms("_train", ["specaugment"])
    writer.flush()


def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


def filter_manifest_df(
    df, is_train_split=False, extra_filters=None, min_n_frames=5, max_n_frames=3000
):
    filters = {
        "no speech": df["audio"] == "",
        f"short speech (<{min_n_frames} frames)": df["n_frames"] < min_n_frames,
        "empty sentence": df["tgt_text"] == "",
    }
    if is_train_split:
        filters[f"long speech (>{max_n_frames} frames)"] = df["n_frames"] > max_n_frames
    if extra_filters is not None:
        filters.update(extra_filters)
    invalid = reduce(lambda x, y: x | y, filters.values())
    valid = ~invalid
    print(
        "| "
        + ", ".join(f"{n}: {f.sum()}" for n, f in filters.items())
        + f", total {invalid.sum()} filtered, {valid.sum()} remained."
    )
    return df[valid]


class S2TDataConfigWriter(object):
    DEFAULT_VOCAB_FILENAME = "dict.txt"
    DEFAULT_INPUT_FEAT_PER_CHANNEL = 80
    DEFAULT_INPUT_CHANNELS = 1

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for S2T data config")
        self.yaml = yaml
        self.yaml_path = yaml_path
        self.config = {}

    def flush(self):
        with open(self.yaml_path, "w") as f:
            self.yaml.dump(self.config, f)

    def set_audio_root(self, audio_root=""):
        self.config["audio_root"] = audio_root

    def set_vocab_filename(self, vocab_filename="dict.txt"):
        self.config["vocab_filename"] = vocab_filename

    def set_specaugment(
        self,
        time_wrap_w: int,
        freq_mask_n: int,
        freq_mask_f: int,
        time_mask_n: int,
        time_mask_t: int,
        time_mask_p: float,
    ):
        self.config["specaugment"] = {
            "time_wrap_W": time_wrap_w,
            "freq_mask_N": freq_mask_n,
            "freq_mask_F": freq_mask_f,
            "time_mask_N": time_mask_n,
            "time_mask_T": time_mask_t,
            "time_mask_p": time_mask_p,
        }

    def set_specaugment_lb_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=1,
            freq_mask_f=27,
            time_mask_n=1,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_specaugment_ld_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_input_channels(self, input_channels=1):
        self.config["input_channels"] = input_channels

    def set_input_feat_per_channel(self, input_feat_per_channel=80):
        self.config["input_feat_per_channel"] = input_feat_per_channel

    def set_bpe_tokenizer(self, bpe_tokenizer: Dict[str, Any]):
        self.config["bpe_tokenizer"] = bpe_tokenizer

    def set_feature_transforms(self, split, transforms: List[str]):
        if "transforms" not in self.config:
            self.config["transforms"] = {}
        self.config["transforms"][split] = transforms
