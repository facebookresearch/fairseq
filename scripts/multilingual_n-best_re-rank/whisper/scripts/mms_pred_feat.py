import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--preds', type=str)    # from mms
    parser.add_argument('--topk_lids', type=str)     # from whisper
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    preds = [eval(x) for x in open(args.preds, "r").readlines()]
    preds_dict = []
    for p in preds:
        tmp = {}
        for (l,v) in p:
            tmp[l] = v
        preds_dict.append(tmp)
    topk_lids = [x.strip() for x in open(args.topk_lids, "r").readlines()]
    assert args.k * len(preds) == len(topk_lids)

    feat = []
    for i, l in enumerate(topk_lids):
        if l in preds_dict[i // args.k]:
            feat.append(preds_dict[i // args.k][l])
        else:
            feat.append(0)

    with open(args.topk_lids + ".mms_score", "w") as f:
        f.writelines([str(x) + "\n" for x in feat])