import argparse
import json
from collections import defaultdict
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--asr', type=str)  # hypo.score
    parser.add_argument('--ref_lid', type=str)
    parser.add_argument('--topk_lid', type=str)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    ref_lid = [x.strip() for x in open(args.ref_lid, "r").readlines()]
    topk_lid = [x.strip() for x in open(args.topk_lid, "r").readlines()]

    asr = [x.strip() for x in open(args.asr, "r").readlines()]
    asr_new = []
    for x in asr:
        if x == "":
            asr_new.append(-1000)
        else:
            asr_new.append(float(x))

    assert len(asr) == len(topk_lid)
    assert len(asr) == args.k * len(ref_lid)

    confs = defaultdict(list)
    corr_confs = defaultdict(list)
    err_confs = defaultdict(list)

    for i in range(len(asr_new)):
        ref = ref_lid[i // args.k]
        confs[topk_lid[i]].append(asr_new[i])
        if ref == topk_lid[i]:
            corr_confs[topk_lid[i]].append(asr_new[i])
        else:
            err_confs[topk_lid[i]].append(asr_new[i])
    # import pdb;pdb.set_trace()

    for lang in confs:
        print(lang, np.array(confs[lang]).mean(), np.array(corr_confs[lang]).mean(), np.array(err_confs[lang]).mean())

    # mean_conf = {}
    # for lang in confs:
    #     mean_conf[lang] = np.array(confs[lang]).mean()
    # mean_conf_as_feat = []
    # for lang in topk_lid:
    #     mean_conf_as_feat.append(mean_conf[lang])
    # with open(args.asr + ".mean_conf", "w") as f:
    #     f.writelines([str(x) + "\n" for x in mean_conf_as_feat])

    # mean_corr_conf = {}
    # for lang in corr_confs:
    #     val = np.array(corr_confs[lang]).mean()
    #     mean_corr_conf[lang] = val
    # mean_corr_conf_as_feat = []
    # mean_of_means = np.array([mean_corr_conf[x] for x in mean_corr_conf]).mean()
    # for lang in topk_lid:
    #     if lang in mean_corr_conf:
    #         mean_corr_conf_as_feat.append(mean_corr_conf[lang])
    #     else:
    #         mean_corr_conf_as_feat.append(mean_of_means)
    # with open(args.asr + ".mean_corr_conf", "w") as f:
    #     f.writelines([str(x) + "\n" for x in mean_corr_conf_as_feat])


    # mean_corr_conf_as_feat_v2 = []  # rand instead of mean
    # list_of_means = [mean_corr_conf[x] for x in mean_corr_conf]
    # for lang in topk_lid:
    #     if lang in mean_corr_conf:
    #         mean_corr_conf_as_feat_v2.append(mean_corr_conf[lang])
    #     else:
    #         mean_corr_conf_as_feat_v2.append(random.choice(list_of_means))
    # with open(args.asr + ".mean_corr_conf_vrand", "w") as f:
    #     f.writelines([str(x) + "\n" for x in mean_corr_conf_as_feat_v2])