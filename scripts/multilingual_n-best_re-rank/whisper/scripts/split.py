import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--wavs', type=str)
    parser.add_argument('--lids', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--splits', type=int)
    args = parser.parse_args()

    wavs = open(args.wavs, "r").readlines()
    lids = open(args.lids, "r").readlines()
    assert len(wavs) == len(lids)

    wavs_splits = []
    lids_splits = []
    n_per = len(wavs) // args.splits
    for i in range(args.splits):
        start_idx = i * n_per
        end_idx = (i + 1) * n_per
        wavs_splits.append(wavs[start_idx:end_idx])
        lids_splits.append(lids[start_idx:end_idx])
        
    if end_idx < len(wavs):
        wavs_splits[-1] += wavs[end_idx:]
        lids_splits[-1] += lids[end_idx:]
        # lids_splits.append(lids[start_idx:end_idx])

    for i in range(args.splits):
        if not os.path.exists(args.dst + "/split_" + str(i)):
            os.makedirs(args.dst + "/split_" + str(i))
        with open(args.dst + "/split_" + str(i) + "/wav.txt", "w") as f:
            f.writelines(wavs_splits[i])
        with open(args.dst + "/split_" + str(i) + "/lid.txt", "w") as f:
            f.writelines(lids_splits[i])
        
        print(f"python scripts/infer_asr.py --wavs {args.dst}/split_{str(i)}/wav.txt --lids {args.dst}/split_{str(i)}/lid.txt --dst {args.exp}/split_{str(i)} --model large-v2")