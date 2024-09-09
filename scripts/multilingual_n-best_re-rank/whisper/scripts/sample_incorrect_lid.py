import argparse
import json
from collections import defaultdict
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--ref', type=str)
    parser.add_argument('--langs', type=str)
    parser.add_argument('--tag', type=str, default="")
    args = parser.parse_args()

    ref = [x.strip() for x in open(args.ref, "r").readlines()]
    langs = [x.strip() for x in open(args.langs, "r").readlines()]

    sampled = []
    for r in ref:
        s = r
        while s == r:
            s = random.sample(langs, 1)[0]
        sampled.append(s+"\n")
    if args.tag == "":
        tag = ""
    else:
        tag = "." + args.tag
    with open(args.ref + ".sampled" + tag, "w") as f:
        f.writelines(sampled)