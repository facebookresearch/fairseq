import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--file', type=str)
    args = parser.parse_args()

    srcs = [x.split("_")[0] for x in open(args.file, "r").readlines()]

    stats = defaultdict(int)

    for s in srcs:
        stats[s] += 1

    with open(args.file + ".stats", "w") as f:
        for l in stats.keys():
            f.write(l + "\t" + str(stats[l]) + "\n")