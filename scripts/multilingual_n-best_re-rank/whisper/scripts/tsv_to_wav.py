import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--tsv', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    lines = open(args.tsv, "r").readlines()
    prefix = lines[0].strip()
    new_lines = []
    for l in lines[1:]:
        pth = prefix + "/" + l.split()[0]
        new_lines.append(pth + "\n")

    with open(args.dst + "/wav.txt", "w") as f:
        f.writelines(new_lines)