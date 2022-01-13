# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from collections import Counter, defaultdict

import pandas as pd
import torchaudio
from tqdm import tqdm

from fairseq.data.audio.audio_utils import convert_waveform
from examples.speech_to_text.data_utils import (
    create_zip,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_tsv_to_dicts,
    save_df_to_tsv
)
from examples.speech_synthesis.data_utils import (
    extract_logmel_spectrogram, extract_pitch, extract_energy, get_global_cmvn,
    ipa_phonemize, get_mfa_alignment, get_unit_alignment,
    get_feature_value_min_max
)


log = logging.getLogger(__name__)


def process(args):
    assert "train" in args.splits
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)

    print("Fetching data...")
    audio_manifest_root = Path(args.audio_manifest_root).absolute()
    samples = []
    for s in args.splits:
        for e in load_tsv_to_dicts(audio_manifest_root / f"{s}.audio.tsv"):
            e["split"] = s
            samples.append(e)
    sample_ids = [s["id"] for s in samples]

    # Get alignment info
    id_to_alignment = None
    if args.textgrid_zip is not None:
        assert args.id_to_units_tsv is None
        id_to_alignment = get_mfa_alignment(
            args.textgrid_zip, sample_ids, args.sample_rate, args.hop_length
        )
    elif args.id_to_units_tsv is not None:
        # assume identical hop length on the unit sequence
        id_to_alignment = get_unit_alignment(args.id_to_units_tsv, sample_ids)

    # Extract features and pack features into ZIP
    feature_name = "logmelspec80"
    zip_path = out_root / f"{feature_name}.zip"
    pitch_zip_path = out_root / "pitch.zip"
    energy_zip_path = out_root / "energy.zip"
    gcmvn_npz_path = out_root / "gcmvn_stats.npz"
    if zip_path.exists() and gcmvn_npz_path.exists():
        print(f"{zip_path} and {gcmvn_npz_path} exist.")
    else:
        feature_root = out_root / feature_name
        feature_root.mkdir(exist_ok=True)
        pitch_root = out_root / "pitch"
        energy_root = out_root / "energy"
        if args.add_fastspeech_targets:
            pitch_root.mkdir(exist_ok=True)
            energy_root.mkdir(exist_ok=True)
        print("Extracting Mel spectrogram features...")
        for sample in tqdm(samples):
            waveform, sample_rate = torchaudio.load(sample["audio"])
            waveform, sample_rate = convert_waveform(
                waveform, sample_rate, normalize_volume=args.normalize_volume,
                to_sample_rate=args.sample_rate
            )
            sample_id = sample["id"]
            target_length = None
            if id_to_alignment is not None:
                a = id_to_alignment[sample_id]
                target_length = sum(a.frame_durations)
                if a.start_sec is not None and a.end_sec is not None:
                    start_frame = int(a.start_sec * sample_rate)
                    end_frame = int(a.end_sec * sample_rate)
                    waveform = waveform[:, start_frame: end_frame]
            extract_logmel_spectrogram(
                waveform, sample_rate, feature_root / f"{sample_id}.npy",
                win_length=args.win_length, hop_length=args.hop_length,
                n_fft=args.n_fft, n_mels=args.n_mels, f_min=args.f_min,
                f_max=args.f_max, target_length=target_length
            )
            if args.add_fastspeech_targets:
                assert id_to_alignment is not None
                extract_pitch(
                    waveform, sample_rate, pitch_root / f"{sample_id}.npy",
                    hop_length=args.hop_length, log_scale=True,
                    phoneme_durations=id_to_alignment[sample_id].frame_durations
                )
                extract_energy(
                    waveform, energy_root / f"{sample_id}.npy",
                    hop_length=args.hop_length, n_fft=args.n_fft,
                    log_scale=True,
                    phoneme_durations=id_to_alignment[sample_id].frame_durations
                )
        print("ZIPing features...")
        create_zip(feature_root, zip_path)
        get_global_cmvn(feature_root, gcmvn_npz_path)
        shutil.rmtree(feature_root)
        if args.add_fastspeech_targets:
            create_zip(pitch_root, pitch_zip_path)
            shutil.rmtree(pitch_root)
            create_zip(energy_root, energy_zip_path)
            shutil.rmtree(energy_root)

    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    pitch_paths, pitch_lengths, energy_paths, energy_lengths = [None] * 4
    if args.add_fastspeech_targets:
        pitch_paths, pitch_lengths = get_zip_manifest(pitch_zip_path)
        energy_paths, energy_lengths = get_zip_manifest(energy_zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    id_to_cer = None
    if args.cer_threshold is not None:
        assert Path(args.cer_tsv_path).is_file()
        id_to_cer = {
            x["id"]: x["uer"] for x in load_tsv_to_dicts(args.cer_tsv_path)
        }
    manifest_by_split = {split: defaultdict(list) for split in args.splits}
    for sample in tqdm(samples):
        sample_id, split = sample["id"], sample["split"]

        if args.snr_threshold is not None and "snr" in sample \
                and sample["snr"] < args.snr_threshold:
            continue
        if args.cer_threshold is not None \
                and id_to_cer[sample_id] > args.cer_threhold:
            continue

        normalized_utt = sample["tgt_text"]
        if id_to_alignment is not None:
            normalized_utt = " ".join(id_to_alignment[sample_id].tokens)
        elif args.ipa_vocab:
            normalized_utt = ipa_phonemize(
                normalized_utt, lang=args.lang, use_g2p=args.use_g2p
            )
        manifest_by_split[split]["id"].append(sample_id)
        manifest_by_split[split]["audio"].append(audio_paths[sample_id])
        manifest_by_split[split]["n_frames"].append(audio_lengths[sample_id])
        manifest_by_split[split]["tgt_text"].append(normalized_utt)
        manifest_by_split[split]["speaker"].append(sample["speaker"])
        manifest_by_split[split]["src_text"].append(sample["src_text"])
        if args.add_fastspeech_targets:
            assert id_to_alignment is not None
            duration = " ".join(
                str(d) for d in id_to_alignment[sample_id].frame_durations
            )
            manifest_by_split[split]["duration"].append(duration)
            manifest_by_split[split]["pitch"].append(pitch_paths[sample_id])
            manifest_by_split[split]["energy"].append(energy_paths[sample_id])
    for split in args.splits:
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest_by_split[split]),
            out_root / f"{split}.tsv"
        )
    # Generate vocab
    vocab_name, spm_filename = None, None
    if id_to_alignment is not None or args.ipa_vocab:
        vocab = Counter()
        for t in manifest_by_split["train"]["tgt_text"]:
            vocab.update(t.split(" "))
        vocab_name = "vocab.txt"
        with open(out_root / vocab_name, "w") as f:
            for s, c in vocab.most_common():
                f.write(f"{s} {c}\n")
    else:
        spm_filename_prefix = "spm_char"
        spm_filename = f"{spm_filename_prefix}.model"
        with NamedTemporaryFile(mode="w") as f:
            for t in manifest_by_split["train"]["tgt_text"]:
                f.write(t + "\n")
            f.flush()  # needed to ensure gen_vocab sees dumped text
            gen_vocab(Path(f.name), out_root / spm_filename_prefix, "char")
    # Generate speaker list
    speakers = sorted({sample["speaker"] for sample in samples})
    speakers_path = out_root / "speakers.txt"
    with open(speakers_path, "w") as f:
        for speaker in speakers:
            f.write(f"{speaker}\n")
    # Generate config YAML
    win_len_t = args.win_length / args.sample_rate
    hop_len_t = args.hop_length / args.sample_rate
    extra = {
        "sample_rate": args.sample_rate,
        "features": {
            "type": "spectrogram+melscale+log",
            "eps": 1e-5, "n_mels": args.n_mels, "n_fft": args.n_fft,
            "window_fn": "hann", "win_length": args.win_length,
            "hop_length": args.hop_length, "sample_rate": args.sample_rate,
            "win_len_t": win_len_t, "hop_len_t": hop_len_t,
            "f_min": args.f_min, "f_max": args.f_max,
            "n_stft": args.n_fft // 2 + 1
        }
    }
    if len(speakers) > 1:
        extra["speaker_set_filename"] = "speakers.txt"
    if args.add_fastspeech_targets:
        pitch_min, pitch_max = get_feature_value_min_max(
            [(out_root / n).as_posix() for n in pitch_paths.values()]
        )
        energy_min, energy_max = get_feature_value_min_max(
            [(out_root / n).as_posix() for n in energy_paths.values()]
        )
        extra["features"]["pitch_min"] = pitch_min
        extra["features"]["pitch_max"] = pitch_max
        extra["features"]["energy_min"] = energy_min
        extra["features"]["energy_max"] = energy_max
    gen_config_yaml(
        out_root, spm_filename=spm_filename, vocab_name=vocab_name,
        audio_root=out_root.as_posix(), input_channels=None,
        input_feat_per_channel=None, specaugment_policy=None,
        cmvn_type="global", gcmvn_path=gcmvn_npz_path, extra=extra
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-manifest-root", "-m", required=True, type=str)
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--splits", "-s", type=str, nargs="+",
                        default=["train", "dev", "test"])
    parser.add_argument("--ipa-vocab", action="store_true")
    parser.add_argument("--use-g2p", action="store_true")
    parser.add_argument("--lang", type=str, default="en-us")
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--f-min", type=int, default=20)
    parser.add_argument("--f-max", type=int, default=8000)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--normalize-volume", "-n", action="store_true")
    parser.add_argument("--textgrid-zip", type=str, default=None)
    parser.add_argument("--id-to-units-tsv", type=str, default=None)
    parser.add_argument("--add-fastspeech-targets", action="store_true")
    parser.add_argument("--snr-threshold", type=float, default=None)
    parser.add_argument("--cer-threshold", type=float, default=None)
    parser.add_argument("--cer-tsv-path", type=str, default="")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
