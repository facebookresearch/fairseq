import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--lid', type=str)
    parser.add_argument('--mapping', type=str)
    args = parser.parse_args()

    lid = [x.strip() for x in open(args.lid, "r").readlines()]
    # whisper_lid_code:mms_lid_code
    mapping = {x[0]:x[1] for x in [l.strip().split(":", 1) for l in open(args.mapping, "r").readlines()]}
    
    lid2 = [mapping[x] + "\n" for x in lid]

    with open(args.lid + "2", "w") as f:
        f.writelines(lid2)