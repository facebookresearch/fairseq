import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import re
import soundfile as sf
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--wavs', type=str)
    parser.add_argument('--refs', type=str)
    parser.add_argument('--hyps1', type=str)    #ora
    parser.add_argument('--hyps2', type=str)    #lid-dep
    parser.add_argument('--lids', type=str)
    parser.add_argument('--confusions', type=str)
    parser.add_argument('--include', nargs="*", default=[])
    parser.add_argument('--dst', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--set', type=str)
    args = parser.parse_args()

    # manifest format: {"audio_filepath": "path/to/audio.wav", "duration": 3.45, "text": "this is a nemo tutorial"}
    wavs = open(args.wavs, "r").readlines()
    refs = open(args.refs, "r").readlines()
    hyps1 = open(args.hyps1, "r").readlines()
    hyps2 = open(args.hyps2, "r").readlines()
    lids = open(args.lids, "r").readlines()
    confusions = open(args.confusions, "r").readlines()

    assert len(wavs) == len(refs)
    assert len(wavs) == len(hyps1)
    assert len(wavs) == len(hyps2)
    assert len(wavs) == len(lids)

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    with open(args.dst + "/manifest.json", "w") as f1:
        for w, r, h1, h2, l, c in tqdm(zip(wavs, refs, hyps1, hyps2, lids, confusions)):
            audio, samplerate = sf.read(w.strip())
            duration = len(audio) / samplerate
            r = r.replace('"', "").lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
            h1 = h1.replace('"', "").lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
            h2 = h2.replace('"', "").lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
            
            # skip empty refs
            if r == "":
                continue
            if r.strip() == "":
                continue
            if len(r) == 0:
                continue

            wer1 = editdistance.eval(h1.split(), r.split()) / len(r.split())
            wer2 = editdistance.eval(h2.split(), r.split()) / len(r.split())
            wer_delta_abs = wer2 - wer1
            if wer1 == 0:
                # wer_delta_rel = '\"inf\"'
                # wer_delta_rel = "float('inf')"
                wer_delta_rel = str(99999999999)
            else:
                wer_delta_rel = f"{(wer_delta_abs / wer1):.2f}"
            f1.write(f"{{\"audio_filepath\": \"{w.strip()}\", \"model\": \"{args.model}\", \"set\": \"{args.set}\", \"oracle_WER\": {wer1:.2f}, \"lid-dep_WER\": {wer2:.2f}, \"WER_abs_delta\": {wer_delta_abs:.2f}, \"WER_rel_delta\": {wer_delta_rel}, \"duration\": {duration}, \"lang\": \"{l.strip()}\", \"confusion\": \"{c.strip()}\", \"text\": \"{r.strip()}\", \"pred_text_oracle\": \"{h1.strip()}\", \"pred_text_lid-dep\": \"{h2.strip()}\"}}\n")
            # f2.write(h1)
            # f3.write(h2)

