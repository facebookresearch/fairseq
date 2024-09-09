import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--tsv', type=str)    # data/*.html
    parser.add_argument('--dst', type=str)
    parser.add_argument('--splits', type=int)
    args = parser.parse_args()

    data = open(args.tsv, "r").readlines()
    first = data[0]
    rest = data[1:]

    splits = []
    n_per = len(rest) // args.splits
    for i in range(args.splits):
        start_idx = i * n_per
        end_idx = (i + 1) * n_per
        splits.append(rest[start_idx:end_idx])
    if end_idx < len(rest):
        splits[-1] += rest[end_idx:]

    assert len(rest) == sum([len(x) for x in splits])

    for i in range(args.splits):
        if not os.path.exists(args.dst + "/split_" + str(i)):
            os.makedirs(args.dst + "/split_" + str(i))
        with open(args.dst + "/split_" + str(i) + "/split.tsv", "w") as f:
            f.write(first)
            f.writelines(splits[i])
        with open(args.dst + "/split_" + str(i) + "/split.label", "w") as f:
            f.writelines(["dummy\n" for _ in range(len(splits[i]))]) #dummy file