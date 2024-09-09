import argparse
import json
from collections import defaultdict
import os
import numpy as np
import editdistance
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--txt', type=str)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    txt = [x.strip() for x in open(args.txt, "r").readlines()]
    assert len(txt) % args.k == 0

    scores = []
    for i in tqdm(range(len(txt) // args.k)):
        start_idx = i
        end_idx = start_idx + args.k
        cands = txt[start_idx:end_idx]

        for cand_idx, c in enumerate(cands):
            tmp = []
            for ref_idx, r in enumerate(cands):
                if cand_idx == ref_idx:
                    continue
                if r == "":
                    continue
                c_words = c.split()
                r_words = r.split()
                errs = editdistance.eval(c_words, r_words)
                err_rate = errs / len(r_words)
                tmp.append(err_rate)
            scores.append(np.array(tmp).mean())
    
    assert len(scores) == len(txt)

    with open(args.txt + ".score", "w") as f:
        f.writelines([str(x) + "\n" for x in scores])