import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--ref', type=str)
    args = parser.parse_args()

    langs = set()
    refs = open(args.ref, "r").readlines()

    for r in refs:
        langs.add(r)

    with open(args.ref + ".set", "w") as f:
        f.writelines(langs)