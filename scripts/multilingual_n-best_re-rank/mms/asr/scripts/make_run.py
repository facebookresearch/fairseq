import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--wavs', type=str)
    parser.add_argument('--lids', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    wavs = [x.strip() for x in open(args.wavs, "r").readlines()]
    lids = [x.strip() for x in open(args.lids, "r").readlines()]

    assert len(wavs) == len(lids)

    with open(args.dst + "/run.sh", "w") as f:
        for w,l in zip(wavs,lids):
            cmd = f"python examples/mms/asr/infer/mms_infer.py --model {args.model} --lang {l} --audio {w} >> {args.dst}/predictions.txt"
            f.write(cmd+"\n")