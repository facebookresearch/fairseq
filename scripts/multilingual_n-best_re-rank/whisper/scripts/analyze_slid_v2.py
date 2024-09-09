# oracle lid version
import argparse
import json
from collections import defaultdict
import numpy as np
import random
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--pred', type=str)
    parser.add_argument('--ref_lid', type=str)
    parser.add_argument('--topk_lid', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    pred = [x.strip() for x in open(args.pred, "r").readlines()]
    ref_lid = [x.strip() for x in open(args.ref_lid, "r").readlines()]

    slid_new = []
    slid_label = []
    for x in pred:
        data = eval(x)
        for y in data:
            slid_new.append(math.log(y[1]))
            slid_label.append(y[0])

    confs = defaultdict(list)   # all
    confs_corr = defaultdict(list)
    confs_err = defaultdict(list)

    for i in range(len(slid_new)):
        confs[slid_label[i]].append(slid_new[i])
        if slid_label[i] == ref_lid[i // 10]:
            confs_corr[slid_label[i]].append(slid_new[i])
        else:
            confs_err[slid_label[i]].append(slid_new[i])

    # for lang in confs:
    #     print(lang, np.array(confs[lang]).mean(), np.array(confs_corr[lang]).mean(), np.array(confs_err[lang]).mean())

    mean_conf = {}
    # for lang in confs_corr:
    #     mean_conf[lang] = np.array(confs_corr[lang]).mean()
    for lang in confs:
        mean_conf[lang] = np.array(confs[lang]).mean()

    mean_of_means = np.array([mean_conf[x] for x in mean_conf]).mean()
    # # import pdb;pdb.set_trace()
    topk_lid = [x.strip() for x in open(args.topk_lid, "r").readlines()]
    mean_conf_as_feat = []
    for lang in topk_lid:
        if lang in mean_conf:
            mean_conf_as_feat.append(mean_conf[lang])
        else:
            mean_conf_as_feat.append(mean_of_means)
    with open(args.dst, "w") as f:
        f.writelines([str(x) + "\n" for x in mean_conf_as_feat])