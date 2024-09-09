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
    parser.add_argument('--units', type=str)
    args = parser.parse_args()

    lines = [x.strip() for x in open(args.units, "r").readlines()]

    lengths = []
    for x in lines:
        l = len(x.split(" "))
        lengths.append(l)
        # import pdb;pdb.set_trace()

    with open(args.units + ".length", "w") as f:
        f.writelines([str(x)+"\n" for x in lengths])