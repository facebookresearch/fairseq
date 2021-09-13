# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
from pathlib import Path
import zipfile
from functools import reduce
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Union
import io

import numpy as np
import pandas as pd
import sentencepiece as sp
from fairseq.data.audio.audio_utils import (
    convert_waveform, _get_kaldi_fbank, _get_torchaudio_fbank, is_npy_data,
    is_sf_audio_data
)
import torch
import soundfile as sf
from tqdm import tqdm


UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1


def gen_vocab(
    input_path: Path, output_path_prefix: Path, model_type="bpe",
    vocab_size=1000, special_symbols: Optional[List[str]] = None
):
    # Train SentencePiece Model
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={output_path_prefix.as_posix()}",
        f"--model_type={model_type}",
        f"--vocab_size={vocab_size}",
        "--character_coverage=1.0",
        f"--num_threads={cpu_count()}",
        f"--unk_id={UNK_TOKEN_ID}",
        f"--bos_id={BOS_TOKEN_ID}",
        f"--eos_id={EOS_TOKEN_ID}",
        f"--pad_id={PAD_TOKEN_ID}",
    ]
    if special_symbols is not None:
        _special_symbols = ",".join(special_symbols)
        arguments.append(f"--user_defined_symbols={_special_symbols}")
    sp.SentencePieceTrainer.Train(" ".join(arguments))
    # Export fairseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix.as_posix() + ".model")
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
    with open(output_path_prefix.as_posix() + ".txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")


def extract_fbank_features(
    waveform: torch.FloatTensor,
    sample_rate: int,
    output_path: Optional[Path] = None,
    n_mel_bins: int = 80,
    overwrite: bool = False,
):
    if output_path is not None and output_path.is_file() and not overwrite:
        return

    _waveform = convert_waveform(waveform, sample_rate, to_mono=True)
    # Kaldi compliance: 16-bit signed integers
    _waveform = _waveform * (2 ** 15)
    _waveform = _waveform.numpy()

    features = _get_kaldi_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable fbank feature extraction"
        )

    if output_path is not None:
        np.save(output_path.as_posix(), features)
    return features


def create_zip(data_root: Path, zip_path: Path):
    paths = list(data_root.glob("*.npy"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for path in tqdm(paths):
            f.write(path, arcname=path.name)


def get_zip_manifest(
        zip_path: Path, zip_root: Optional[Path] = None, is_audio=False
):
    _zip_path = Path.joinpath(zip_root or Path(""), zip_path)
    with zipfile.ZipFile(_zip_path, mode="r") as f:
        info = f.infolist()
    paths, lengths = {}, {}
    for i in tqdm(info):
        utt_id = Path(i.filename).stem
        offset, file_size = i.header_offset + 30 + len(i.filename), i.file_size
        paths[utt_id] = f"{zip_path.as_posix()}:{offset}:{file_size}"
        with open(_zip_path, "rb") as f:
            f.seek(offset)
            byte_data = f.read(file_size)
            assert len(byte_data) > 1
            if is_audio:
                assert is_sf_audio_data(byte_data), i
            else:
                assert is_npy_data(byte_data), i
            byte_data_fp = io.BytesIO(byte_data)
            if is_audio:
                lengths[utt_id] = sf.info(byte_data_fp).frames
            else:
                lengths[utt_id] = np.load(byte_data_fp).shape[0]
    return paths, lengths


def gen_config_yaml(
    manifest_root: Path,
    spm_filename: Optional[str] = None,
    vocab_name: Optional[str] = None,
    yaml_filename: str = "config.yaml",
    specaugment_policy: Optional[str] = "lb",
    prepend_tgt_lang_tag: bool = False,
    sampling_alpha: Optional[float] = None,
    input_channels: Optional[int] = 1,
    input_feat_per_channel: Optional[int] = 80,
    audio_root: str = "",
    cmvn_type: str = "utterance",
    gcmvn_path: Optional[Path] = None,
    extra=None
):
    manifest_root = manifest_root.absolute()
    writer = S2TDataConfigWriter(manifest_root / yaml_filename)
    assert spm_filename is not None or vocab_name is not None
    vocab_name = spm_filename.replace(".model", ".txt") if vocab_name is None \
        else vocab_name
    writer.set_vocab_filename(vocab_name)
    if input_channels is not None:
        writer.set_input_channels(input_channels)
    if input_feat_per_channel is not None:
        writer.set_input_feat_per_channel(input_feat_per_channel)
    specaugment_setters = {
        "lb": writer.set_specaugment_lb_policy,
        "ld": writer.set_specaugment_ld_policy,
        "sm": writer.set_specaugment_sm_policy,
        "ss": writer.set_specaugment_ss_policy,
    }
    specaugment_setter = specaugment_setters.get(specaugment_policy, None)
    if specaugment_setter is not None:
        specaugment_setter()
    if spm_filename is not None:
        writer.set_bpe_tokenizer(
            {
                "bpe": "sentencepiece",
                "sentencepiece_model": (manifest_root / spm_filename).as_posix(),
            }
        )
    if prepend_tgt_lang_tag:
        writer.set_prepend_tgt_lang_tag(True)
    if sampling_alpha is not None:
        writer.set_sampling_alpha(sampling_alpha)

    if cmvn_type not in ["global", "utterance"]:
        raise NotImplementedError

    if specaugment_policy is not None:
        writer.set_feature_transforms(
            "_train", [f"{cmvn_type}_cmvn", "specaugment"]
        )
    writer.set_feature_transforms("*", [f"{cmvn_type}_cmvn"])

    if cmvn_type == "global":
        if gcmvn_path is None:
            raise ValueError("Please provide path of global cmvn file.")
        else:
            writer.set_global_cmvn(gcmvn_path.as_posix())

    if len(audio_root) > 0:
        writer.set_audio_root(audio_root)

    if extra is not None:
        writer.set_extra(extra)
    writer.flush()


def load_df_from_tsv(path: Union[str, Path]) -> pd.DataFrame:
    _path = path if isinstance(path, str) else path.as_posix()
    return pd.read_csv(
        _path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )


def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


def load_tsv_to_dicts(path: Union[str, Path]) -> List[dict]:
    with open(path, "r") as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        rows = [dict(e) for e in reader]
    return rows


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


def cal_gcmvn_stats(features_list):
    features = np.concatenate(features_list)
    square_sums = (features ** 2).sum(axis=0)
    mean = features.mean(axis=0)
    features = np.subtract(features, mean)
    var = square_sums / features.shape[0] - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    return {"mean": mean.astype("float32"), "std": std.astype("float32")}


class S2TDataConfigWriter(object):
    DEFAULT_VOCAB_FILENAME = "dict.txt"
    DEFAULT_INPUT_FEAT_PER_CHANNEL = 80
    DEFAULT_INPUT_CHANNELS = 1

    def __init__(self, yaml_path: Path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML for S2T data config YAML files")
        self.yaml = yaml
        self.yaml_path = yaml_path
        self.config = {}

    def flush(self):
        with open(self.yaml_path, "w") as f:
            self.yaml.dump(self.config, f)

    def set_audio_root(self, audio_root=""):
        self.config["audio_root"] = audio_root

    def set_vocab_filename(self, vocab_filename: str = "dict.txt"):
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

    def set_specaugment_sm_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=15,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )

    def set_specaugment_ss_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )

    def set_input_channels(self, input_channels: int = 1):
        self.config["input_channels"] = input_channels

    def set_input_feat_per_channel(self, input_feat_per_channel: int = 80):
        self.config["input_feat_per_channel"] = input_feat_per_channel

    def set_bpe_tokenizer(self, bpe_tokenizer: Dict[str, Any]):
        self.config["bpe_tokenizer"] = bpe_tokenizer

    def set_global_cmvn(self, stats_npz_path: str):
        self.config["global_cmvn"] = {"stats_npz_path": stats_npz_path}

    def set_feature_transforms(self, split: str, transforms: List[str]):
        if "transforms" not in self.config:
            self.config["transforms"] = {}
        self.config["transforms"][split] = transforms

    def set_prepend_tgt_lang_tag(self, flag: bool = True):
        self.config["prepend_tgt_lang_tag"] = flag

    def set_sampling_alpha(self, sampling_alpha: float = 1.0):
        self.config["sampling_alpha"] = sampling_alpha

    def set_extra(self, data):
        self.config.update(data)
