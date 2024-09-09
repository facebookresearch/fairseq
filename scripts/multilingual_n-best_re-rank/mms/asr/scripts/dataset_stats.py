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
    parser.add_argument('--langs', type=str)
    args = parser.parse_args()

    langs = [x.strip() for x in open(args.langs, "r").readlines()]

    stats = defaultdict(int)

    for l in langs:
        stats[l] += 1

    with open(args.langs + ".stats", "w") as f:
        for l in stats.keys():
            f.write(l + "\t" + str(stats[l]) + "\n")