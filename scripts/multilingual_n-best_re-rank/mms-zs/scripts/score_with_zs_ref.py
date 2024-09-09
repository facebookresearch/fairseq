import argparse
import json
from collections import defaultdict
import os
import numpy as np
import editdistance
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--hyp', type=str)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    hyp = [x.strip() for x in open(args.hyp, "r").readlines()]
    ref = [x.strip() for x in open(args.ref, "r").readlines()]
    assert len(hyp) % args.k == 0
    assert len(hyp) == args.k * len(ref)

    scores = []
    for i in tqdm(range(len(ref))):
        start_idx = i
        end_idx = start_idx + args.k
        cands = hyp[start_idx:end_idx]

        for cand_idx, c in enumerate(cands):
            c_words = c.split()
            r_words = ref[i].split()
            if len(r_words) == 0:
                scores.append(0)
                continue
            errs = editdistance.eval(c_words, r_words)
            err_rate = errs / len(r_words)
            scores.append(err_rate)
    
    assert len(scores) == len(hyp)

    with open(args.hyp + ".zs_ref_score", "w") as f:
        f.writelines([str(x) + "\n" for x in scores])