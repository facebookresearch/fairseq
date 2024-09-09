import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--txt', type=str)
    parser.add_argument('--lid', type=str)
    parser.add_argument('--p', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--splits', type=int)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    txt = open(args.txt, "r").readlines()
    lid = open(args.lid, "r").readlines()
    p = open(args.p, "r").readlines()
    assert len(txt) == len(p)
    assert len(txt) == len(lid)

    txt_splits = []
    lid_splits = []
    p_splits = []
    n_per = len(txt) // args.splits
    for i in range(args.splits):
        start_idx = i * n_per
        end_idx = (i + 1) * n_per
        txt_splits.append(txt[start_idx:end_idx])
        lid_splits.append(lid[start_idx:end_idx])
        p_splits.append(p[start_idx:end_idx])
        
    if end_idx_txt < len(txt):
        txt_splits[-1] += txt[end_idx:]
        lid_splits[-1] += lid[end_idx:]
        p_splits[-1] += p[end_idx:]

    for i in range(args.splits):
        if not os.path.exists(args.dst + "/split_" + str(i)):
            os.makedirs(args.dst + "/split_" + str(i))
        with open(args.dst + "/split_" + str(i) + "/txt", "w") as f:
            f.writelines(txt_splits[i])
        with open(args.dst + "/split_" + str(i) + "/lid", "w") as f:
            f.writelines(lid_splits[i])
        with open(args.dst + "/split_" + str(i) + "/p", "w") as f:
            f.writelines(p_splits[i])
        
        print(f"python scripts/falign.py --txt {args.dst}/split_{str(i)}/txt --lid {args.dst}/split_{str(i)}/lid --p {args.dst}/split_{str(i)}/p --dst {args.exp}/split_{str(i)}")