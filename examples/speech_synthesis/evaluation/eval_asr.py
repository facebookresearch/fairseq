# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import editdistance
import re
import shutil
import soundfile as sf
import subprocess
from pathlib import Path

from examples.speech_to_text.data_utils import load_tsv_to_dicts


def preprocess_text(text):
    text = "|".join(re.sub(r"[^A-Z' ]", " ", text.upper()).split())
    text = " ".join(text)
    return text


def prepare_w2v_data(
        dict_dir, sample_rate, label, audio_paths, texts, split, data_dir
):
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        dict_dir / f"dict.{label}.txt",
        data_dir / f"dict.{label}.txt"
    )
    with open(data_dir / f"{split}.tsv", "w") as f:
        f.write("/\n")
        for audio_path in audio_paths:
            wav, sr = sf.read(audio_path)
            assert sr == sample_rate, f"{sr} != sample_rate"
            nsample = len(wav)
            f.write(f"{audio_path}\t{nsample}\n")
    with open(data_dir / f"{split}.{label}", "w") as f:
        for text in texts:
            text = preprocess_text(text)
            f.write(f"{text}\n")


def run_asr(asr_dir, split, w2v_ckpt, w2v_label, res_dir):
    """
    results will be saved at
    {res_dir}/{ref,hypo}.word-{w2v_ckpt.filename}-{split}.txt
    """
    cmd = ["python", "-m", "examples.speech_recognition.infer"]
    cmd += [str(asr_dir.resolve())]
    cmd += ["--task", "audio_finetuning", "--nbest", "1", "--quiet"]
    cmd += ["--w2l-decoder", "viterbi", "--criterion", "ctc"]
    cmd += ["--post-process", "letter", "--max-tokens", "4000000"]
    cmd += ["--path", str(w2v_ckpt.resolve()), "--labels", w2v_label]
    cmd += ["--gen-subset", split, "--results-path", str(res_dir.resolve())]

    print(f"running cmd:\n{' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def compute_error_rate(hyp_wrd_path, ref_wrd_path, unit="word"):
    """each line is "<text> (None-<index>)" """
    tokenize_line = {
        "word": lambda x: re.sub(r" \(.*\)$", "", x.rstrip()).split(),
        "char": lambda x: list(re.sub(r" \(.*\)$", "", x.rstrip()))
    }.get(unit)
    if tokenize_line is None:
        raise ValueError(f"{unit} not supported")

    inds = [int(re.sub(r"\D*(\d*)\D*", r"\1", line))
            for line in open(hyp_wrd_path)]
    hyps = [tokenize_line(line) for line in open(hyp_wrd_path)]
    refs = [tokenize_line(line) for line in open(ref_wrd_path)]
    assert(len(hyps) == len(refs))
    err_rates = [
        editdistance.eval(hyp, ref) / len(ref) for hyp, ref in zip(hyps, refs)
    ]
    ind_to_err_rates = {i: e for i, e in zip(inds, err_rates)}
    return ind_to_err_rates


def main(args):
    samples = load_tsv_to_dicts(args.raw_manifest)
    ids = [
        sample[args.id_header] if args.id_header else "" for sample in samples
    ]
    audio_paths = [sample[args.audio_header] for sample in samples]
    texts = [sample[args.text_header] for sample in samples]

    prepare_w2v_data(
        args.w2v_dict_dir,
        args.w2v_sample_rate,
        args.w2v_label,
        audio_paths,
        texts,
        args.split,
        args.asr_dir
    )
    run_asr(args.asr_dir, args.split, args.w2v_ckpt, args.w2v_label, args.asr_dir)
    ind_to_err_rates = compute_error_rate(
        args.asr_dir / f"hypo.word-{args.w2v_ckpt.name}-{args.split}.txt",
        args.asr_dir / f"ref.word-{args.w2v_ckpt.name}-{args.split}.txt",
        args.err_unit,
    )

    uer_path = args.asr_dir / f"uer_{args.err_unit}.{args.split}.tsv"
    with open(uer_path, "w") as f:
        f.write("id\taudio\tuer\n")
        for ind, (id_, audio_path) in enumerate(zip(ids, audio_paths)):
            f.write(f"{id_}\t{audio_path}\t{ind_to_err_rates[ind]:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--asr-dir", required=True, type=Path)
    parser.add_argument("--id-header", default="id", type=str)
    parser.add_argument("--audio-header", default="audio", type=str)
    parser.add_argument("--text-header", default="src_text", type=str)
    parser.add_argument("--split", default="raw", type=str)
    parser.add_argument("--w2v-ckpt", required=True, type=Path)
    parser.add_argument("--w2v-dict-dir", required=True, type=Path)
    parser.add_argument("--w2v-sample-rate", default=16000, type=int)
    parser.add_argument("--w2v-label", default="ltr", type=str)
    parser.add_argument("--err-unit", default="word", type=str)
    args = parser.parse_args()

    main(args)
