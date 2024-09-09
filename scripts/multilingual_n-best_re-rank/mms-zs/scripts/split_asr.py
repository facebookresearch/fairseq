import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--wav', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--splits', type=int)
    args = parser.parse_args()

    wav = open(args.wav, "r").readlines()

    wav_splits = []
    n_per = len(wav) // args.splits
    for i in range(args.splits):
        start_idx = i * n_per
        end_idx = (i + 1) * n_per
        wav_splits.append(wav[start_idx:end_idx])
        
    if end_idx < len(wav):
        wav_splits[-1] += wav[end_idx:]

    for i in range(args.splits):
        if not os.path.exists(args.dst + "/split_" + str(i)):
            os.makedirs(args.dst + "/split_" + str(i))
        with open(args.dst + "/split_" + str(i) + "/wav.txt", "w") as f:
            f.writelines(wav_splits[i])
        
        print(f"python infer_zs.py --wavs {args.dst}/split_{str(i)}/wav.txt --dst {args.exp}/split_{str(i)}")