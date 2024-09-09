import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--refs', nargs="*")
    parser.add_argument('--mapping', type=str)
    args = parser.parse_args()

    mapping = {x[1]:x[0] for x in [l.strip().split(":", 1) for l in open(args.mapping, "r").readlines()]}

    for ref in args.refs:
        set_langs = set()
        ref_lines = [x.strip() for x in open(ref, "r").readlines()]
        count = 0
        for r in ref_lines:
            if r in mapping.keys():
                count += 1
            set_langs.add(r)
        print(ref)
        print('{:.4g}'.format(count / len(ref_lines)))
        print(len(set_langs))