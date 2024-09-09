import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--ref', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--mapping', type=str)
    args = parser.parse_args()

    mapping = {x[1]:x[0] for x in [l.strip().split(":", 1) for l in open(args.mapping, "r").readlines()]}

    coverage = []
    set_langs = set()
    ref_lines = [x.strip() for x in open(args.ref, "r").readlines()]
    count = 0
    for r in ref_lines:
        if r in mapping.keys():
            coverage.append("1\n")
        else:
            coverage.append("0\n")
    with open(args.dst, "w") as f:
        f.writelines(coverage)