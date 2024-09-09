import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--src', type=str)
    parser.add_argument('--splits', type=int)
    args = parser.parse_args()

    pron = []
    pron_no_sp = []
    for i in range(args.splits):
        pron += open(f"{args.src}/split_{str(i)}/pron", "r").readlines()
        pron_no_sp += open(f"{args.src}/split_{str(i)}/pron_no_sp", "r").readlines()

    with open(args.src + "/pron", "w") as f:
        f.writelines(pron)

    with open(args.src + "/pron_no_sp", "w") as f:
        f.writelines(pron_no_sp)