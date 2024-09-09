# oracle lid version
import argparse
import json
from collections import defaultdict
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--asr', type=str)  # hypo.score
    parser.add_argument('--lid', type=str)
    parser.add_argument('--topk_lid', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    lid = [x.strip() for x in open(args.lid, "r").readlines()]

    asr = [x.strip() for x in open(args.asr, "r").readlines()]
    asr_new = []
    for x in asr:
        if x == "":
            asr_new.append(-1000)
        else:
            asr_new.append(float(x))

    assert len(asr) == len(lid)

    confs = defaultdict(list)

    for i in range(len(asr_new)):
        confs[lid[i]].append(asr_new[i])

    # for lang in confs:
    #     print(lang, np.array(confs[lang]).mean())

    mean_conf = {}
    for lang in confs:
        mean_conf[lang] = np.array(confs[lang]).mean()
        
    hard_code = {"ita":-14}

    mean_of_means = np.array([mean_conf[x] for x in mean_conf]).mean()
    # print("mean", mean_of_means)

    topk_lid = [x.strip() for x in open(args.topk_lid, "r").readlines()]
    mean_conf_as_feat = []
    not_covered = set()
    for lang in topk_lid:
        if lang in mean_conf:
            mean_conf_as_feat.append(mean_conf[lang])
        elif lang in hard_code:
            mean_conf_as_feat.append(hard_code[lang])
        else:
            mean_conf_as_feat.append(mean_of_means)
            not_covered.add(lang)
    with open(args.dst, "w") as f:
        f.writelines([str(x) + "\n" for x in mean_conf_as_feat])

    print(not_covered)