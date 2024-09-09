import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--txt', type=str)
    parser.add_argument('--wav', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--splits', type=int)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    txt = open(args.txt, "r").readlines()
    wav = open(args.wav, "r").readlines()
    assert len(txt) == args.k * len(wav)

    txt_splits = []
    wav_splits = []
    n_per_wav = len(wav) // args.splits
    n_per_txt = n_per_wav * args.k
    for i in range(args.splits):
        start_idx_txt = i * n_per_txt
        end_idx_txt = (i + 1) * n_per_txt
        txt_splits.append(txt[start_idx_txt:end_idx_txt])

        start_idx_wav = i * n_per_wav
        end_idx_wav = (i + 1) * n_per_wav
        wav_splits.append(wav[start_idx_wav:end_idx_wav])
        
    if end_idx_txt < len(txt):
        txt_splits[-1] += txt[end_idx_txt:]

    if end_idx_wav < len(wav):
        wav_splits[-1] += wav[end_idx_wav:]

    for i in range(args.splits):
        if not os.path.exists(args.dst + "/split_" + str(i)):
            os.makedirs(args.dst + "/split_" + str(i))
        with open(args.dst + "/split_" + str(i) + "/txt", "w") as f:
            f.writelines(txt_splits[i])
        with open(args.dst + "/split_" + str(i) + "/wav", "w") as f:
            f.writelines(wav_splits[i])
        
        print(f"python scripts/falign.py --txt {args.dst}/split_{str(i)}/txt --wav {args.dst}/split_{str(i)}/wav --dst {args.exp}/split_{str(i)}")