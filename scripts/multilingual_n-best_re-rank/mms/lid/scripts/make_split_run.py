import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--dump', type=str)
    parser.add_argument('--splits', type=int)
    parser.add_argument('--set', type=str)
    args = parser.parse_args()

    for i in range(args.splits):
        tsv = args.dump + "/split_" + str(i) + "/split.tsv"
        if not os.path.exists(f"exp/mms1b_l4017/{args.set}/split_{i}"):
            os.makedirs(f"exp/mms1b_l4017/{args.set}/split_{i}")
        print(f"python infer.py /checkpoint/yanb/MMS1_models/4017 --path /checkpoint/yanb/MMS1_models/4017/mms1b_l4017.pt --task audio_classification --infer-manifest {tsv} --output-path exp/mms1b_l4017/{args.set}/split_{i}")