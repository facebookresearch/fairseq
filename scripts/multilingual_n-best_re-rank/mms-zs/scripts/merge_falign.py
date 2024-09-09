import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--src', type=str)
    parser.add_argument('--splits', type=int)
    args = parser.parse_args()

    score = []
    for i in range(args.splits):
        score += open(f"{args.src}/split_{str(i)}/score", "r").readlines()

    with open(args.src + "/score", "w") as f:
        f.writelines(score)