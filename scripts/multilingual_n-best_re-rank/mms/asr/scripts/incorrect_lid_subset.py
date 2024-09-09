import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
import subprocess
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--flags', type=str)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--srcs', nargs='*',)
    parser.add_argument('--match', type=str, default="0")
    args = parser.parse_args()

    flags = [x.strip() for x in open(args.flags, "r").readlines()]
    incorrect_idxs = [i for i,x in enumerate(flags) if x == args.match]

    for src in args.srcs:
        lines =  [x for x in open(src, "r").readlines()]
        assert len(flags) == len(lines)
        if args.tag == "":
            dst = src + ".incorrect-lid"
        else:
            dst = src + ".incorrect-lid" + "_" + args.tag
        with open(dst, "w") as f:
            for idx in incorrect_idxs:
                f.write(lines[idx])