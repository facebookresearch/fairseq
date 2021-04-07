import argparse
import logging
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import pandas as pd
import soundfile as sf
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils import get_waveform


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "duration_ms", "n_frames", "tgt_text",
                    "speaker", "tgt_lang"]

WAV_MESSAGE = \
    'for f in ${EUROPARLST_ROOT}/*/audios/*.m4a; do\n' \
    '    ffmpeg -i $f -ar 16000 -hide_banner -loglevel error "${f%.m4a}.wav"' \
    ' && rm $f\n' \
    'done\n'


class EuroparlST(Dataset):
    """
    Create a Dataset for EuroparlST.
    Each item is a tuple of the form: waveform, sample_rate, source utterance,
    target utterance, speaker_id, target language, utterance_id
    """

    SPLITS = ["train", "train-noisy", "dev", "test"]
    LANGPAIRS = [f"{l1}-{l2}"
                 for l2 in ["en", "fr", "de", "it", "es", "pt", "pl", "ro"]
                 for l1 in ["en", "fr", "de", "it", "es", "pt", "pl", "ro"]
                 if l1 != l2]

    def __init__(self, root: str, lang_pair: str, split: str) -> None:
        assert split in self.SPLITS and lang_pair in self.LANGPAIRS
        src_lang, tgt_lang = lang_pair.split('-')
        _root = Path(root) / src_lang
        wav_root = _root / "audios"
        txt_root = _root / tgt_lang / split
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()

        # Create speaker dictionary
        with open(txt_root / "speeches.lst") as f:
            speeches = [r.strip() for r in f]
        with open(txt_root / "speakers.lst") as f:
            speakers = [r.strip() for r in f]
        assert len(speeches) == len(speakers)
        spk_dict = {
            spe: spk.split('_')[-1] for spe, spk in zip(speeches, speakers)
        }

        # Generate segments dictionary
        with open(txt_root / "segments.lst") as f:
            segments = [{
                'wav': r.split(' ')[0],
                'offset': float(r.split(' ')[1]),
                'duration': float(r.split(' ')[2].strip()) - float(r.split(' ')[1]),
                'speaker_id': spk_dict[r.split(' ')[0]],
            } for r in f]

        # Load source and target utterances
        for _lang in [src_lang, tgt_lang]:
            with open(txt_root / f"segments.{_lang}") as f:
                utts = [r.strip() for r in f]
            assert len(segments) == len(utts)
            for i, u in enumerate(utts):
                segments[i][_lang] = u

        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / (wav_filename + ".wav")
            if not wav_path.exists() and wav_path.with_suffix('.m4a'):
                print(f"You must convert the audio files to WAV format "
                      f"(and resampling to 16 kHz is recommended):\n\n"
                      f"{WAV_MESSAGE}\n")
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment[src_lang],
                        segment[tgt_lang],
                        segment["speaker_id"],
                        tgt_lang,
                        _id,
                    )
                )

    def __getitem__(self, n: int) \
            -> Tuple[torch.Tensor, int, str, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, tgt_lang, \
            utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, tgt_lang, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    src_lang, tgt_lang = args.lang_pair.split('-')
    root = Path(args.data_root).absolute()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    if not args.use_audio_input:
        # Extract features
        feature_root = root / src_lang / f"fbank80_{args.lang_pair}"
        feature_root.mkdir(exist_ok=True)
        for split in EuroparlST.SPLITS:
            print(f"Fetching split {split}...")
            dataset = EuroparlST(root, args.lang_pair, split)
            print("Extracting log mel filter bank features...")
            for waveform, sample_rate, _, _, _, _, utt_id in tqdm(dataset):
                extract_fbank_features(
                    waveform, sample_rate, feature_root / f"{utt_id}.npy"
                )
        # Pack features into ZIP
        zip_path = root / src_lang / f"fbank80_{args.lang_pair}.zip"
        print("ZIPing features...")
        create_zip(feature_root, zip_path)
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(zip_path)
        # Clean up
        shutil.rmtree(feature_root)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = f"{args.lang_pair}_{args.task}"
    for split in EuroparlST.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        if args.prepend_tgt_lang_tag:
            manifest["tgt_lang"] = []
        dataset = EuroparlST(root, args.lang_pair, split)
        for i, (wav, sr, src_utt, tgt_utt, speaker_id, tgt_lang, utt_id) \
                in enumerate(tqdm(dataset)):
            manifest["id"].append(utt_id)
            duration_ms = int(wav.size(1) / sr * 1000)
            manifest["duration_ms"].append(duration_ms)
            if args.use_audio_input:
                wav_filename, offset, n_frames = dataset.data[i][:3]
                manifest["audio"].append(
                    f"{wav_filename}:{offset}:{n_frames}"
                )
                manifest["n_frames"].append(wav.size(1))
            else:
                manifest["audio"].append(zip_manifest[utt_id])
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            manifest["tgt_text"].append(
                tgt_utt if args.task == 'st' else src_utt
            )
            manifest["speaker"].append(speaker_id)
            if args.prepend_tgt_lang_tag:
                manifest["tgt_lang"].append(
                    tgt_lang if args.task == 'st' else src_lang
                )
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(
            df,
            root / src_lang / f"{split}_{task}.tsv"
        )
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = \
        f"spm_{args.vocab_type}{vocab_size_str}_{task}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        special_symbols = [
            f"<lang:{tgt_lang if args.task == 'st' else src_lang}>"
        ] if args.prepend_tgt_lang_tag else None
        gen_vocab(
            Path(f.name),
            root / src_lang / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
            special_symbols=special_symbols,
        )
    # Generate config YAML
    gen_config_yaml(
        root / src_lang ,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{task}.yaml",
        prepend_tgt_lang_tag=args.prepend_tgt_lang_tag,
        specaugment_policy="lb" if not args.use_audio_input else None,
        use_audio_input=args.use_audio_input,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    )
    parser.add_argument("--vocab-size", default=1000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--lang-pair", required=True, type=str)
    parser.add_argument("--use-audio-input", action='store_true',
                        help="Use raw audio, instead of extracting features.")
    parser.add_argument("--prepend-tgt-lang-tag", action='store_true',
                        help="Prepend the target language tag when loading "
                             "target sentences.")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
