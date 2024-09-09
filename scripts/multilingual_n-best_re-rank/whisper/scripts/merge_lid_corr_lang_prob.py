import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--src', type=str)
    parser.add_argument('--splits', type=int)
    args = parser.parse_args()

    preds = []
    for i in range(args.splits):
        preds += open(f"{args.src}/split_{str(i)}/corr_lang_prob", "r").readlines()

    with open(args.src + "corr_lang_prob", "w") as f:
        f.writelines(preds)