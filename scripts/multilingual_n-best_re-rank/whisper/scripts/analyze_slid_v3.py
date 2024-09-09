# oracle lid version
import argparse
import json
from collections import defaultdict
import numpy as np
import random
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--corr_lid_prob', type=str)
    parser.add_argument('--ref_lid', type=str)
    parser.add_argument('--topk_lid', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    corr_lid_prob = [x.strip() for x in open(args.corr_lid_prob, "r").readlines()]
    ref_lid = [x.strip() for x in open(args.ref_lid, "r").readlines()]

    slid_new = []
    for x in corr_lid_prob:
        slid_new.append(math.log(float(x)))

    confs = defaultdict(list)   # all

    for i in range(len(slid_new)):
        confs[ref_lid[i]].append(slid_new[i])
    
    mean_conf = {}
    for lang in confs:
        mean_conf[lang] = np.array(confs[lang]).mean()

    mean_of_means = np.array([mean_conf[x] for x in mean_conf]).mean()

    topk_lid = [x.strip() for x in open(args.topk_lid, "r").readlines()]
    mean_conf_as_feat = []
    for lang in topk_lid:
        if lang in mean_conf:
            mean_conf_as_feat.append(mean_conf[lang])
        else:
            mean_conf_as_feat.append(mean_of_means)
    with open(args.dst, "w") as f:
        f.writelines([str(x) + "\n" for x in mean_conf_as_feat])