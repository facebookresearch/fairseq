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
    parser.add_argument('--ref_lid', type=str)
    parser.add_argument('--topk_lid', type=str)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    ref_lid = [x.strip() for x in open(args.ref_lid, "r").readlines()]
    topk_lid = [x.strip() for x in open(args.topk_lid, "r").readlines()]

    assert len(ref_lid) * args.k == len(topk_lid)

    oracle_flag = []
    for i in range(len(ref_lid)):
        start_idx = i * args.k
        end_idx = start_idx + args.k
        cand_lid = topk_lid[start_idx:end_idx]
        
        try:
            ora_idx = cand_lid.index(ref_lid[i])
        except:
            ora_idx = 0

        flags = [0] * args.k
        flags[ora_idx] = 1
        oracle_flag += flags

    assert len(oracle_flag) == len(topk_lid)
    # import pdb;pdb.set_trace()
    with open(args.topk_lid + ".oracle_flag", "w") as f:
        f.writelines([str(x) + "\n" for x in oracle_flag])