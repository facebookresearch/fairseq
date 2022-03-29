# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import torchaudio
import torchaudio.sox_effects as ta_sox
import yaml
from tqdm import tqdm

from examples.speech_to_text.data_utils import load_tsv_to_dicts
from examples.speech_synthesis.preprocessing.speaker_embedder import SpkrEmbedder


def extract_embedding(audio_path, embedder):
    wav, sr = torchaudio.load(audio_path)  # 2D
    if sr != embedder.RATE:
        wav, sr = ta_sox.apply_effects_tensor(
            wav, sr, [["rate", str(embedder.RATE)]]
        )
    try:
        emb = embedder([wav[0].cuda().float()]).cpu().numpy()
    except RuntimeError:
        emb = None
    return emb


def process(args):
    print("Fetching data...")
    raw_manifest_root = Path(args.raw_manifest_root).absolute()
    samples = [load_tsv_to_dicts(raw_manifest_root / (s + ".tsv"))
               for s in args.splits]
    samples = list(chain(*samples))
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(f"{config['audio_root']}/{config['speaker_set_filename']}") as f:
        speaker_to_id = {r.strip(): i for i, r in enumerate(f)}

    embedder = SpkrEmbedder(args.ckpt).cuda()
    speaker_to_cnt = defaultdict(float)
    speaker_to_emb = defaultdict(float)
    for sample in tqdm(samples, desc="extract emb"):
        emb = extract_embedding(sample["audio"], embedder)
        if emb is not None:
            speaker_to_cnt[sample["speaker"]] += 1
            speaker_to_emb[sample["speaker"]] += emb
    if len(speaker_to_emb) != len(speaker_to_id):
        missed = set(speaker_to_id) - set(speaker_to_emb.keys())
        print(
            f"WARNING: missing embeddings for {len(missed)} speaker:\n{missed}"
        )
    speaker_emb_mat = np.zeros((len(speaker_to_id), len(emb)), float)
    for speaker in speaker_to_emb:
        idx = speaker_to_id[speaker]
        emb = speaker_to_emb[speaker]
        cnt = speaker_to_cnt[speaker]
        speaker_emb_mat[idx, :] = emb / cnt
    speaker_emb_name = "speaker_emb.npy"
    speaker_emb_path = f"{config['audio_root']}/{speaker_emb_name}"
    np.save(speaker_emb_path, speaker_emb_mat)
    config["speaker_emb_filename"] = speaker_emb_name

    with open(args.new_config, "w") as f:
        yaml.dump(config, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest-root", "-m", required=True, type=str)
    parser.add_argument("--splits", "-s", type=str, nargs="+",
                        default=["train"])
    parser.add_argument("--config", "-c", required=True, type=str)
    parser.add_argument("--new-config", "-n", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str,
                        help="speaker embedder checkpoint")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
