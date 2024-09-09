import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
import subprocess
import re
import math
import numpy as np
import editdistance
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from functools import partial
import torch
import matplotlib.pyplot as plt

cer_langs = [x.strip() for x in open("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/cer_langs.txt", "r").readlines()]

def plot(data1, data2, dst):
    # import pdb;pdb.set_trace()
    both = np.array(data1+data2)
    bins1 = np.linspace(both.min(), both.max(), 100)
    bins2 = np.linspace(both.min(), both.max(), 100)
    fig = plt.figure()
    plt.hist(data1, bins1, alpha=0.5, label='Right', weights=np.ones(len(data1)) / len(data1))
    plt.hist(data2, bins2, alpha=0.5, label='Wrong', weights=np.ones(len(data2)) / len(data2))
    plt.legend(loc='upper right')
    # plt.show()
    fig.savefig(dst, dpi=fig.dpi)

def normalize(feat):
    # create a StandardScaler object
    scaler = StandardScaler()
    # fit the scaler to the data
    X = np.array(feat).reshape(-1,1)
    scaler.fit(X)
    # transform the data
    X_normalized = scaler.transform(X)
    return X_normalized.flatten().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--slid', type=str) # predictions.txt
    parser.add_argument('--wlid', type=str) # predictions.txt.scores
    parser.add_argument('--asr', type=str)  # hypo.score
    parser.add_argument('--lm', type=str)  # predictions.txt
    parser.add_argument('--fa_mms', type=str) # falign.score
    parser.add_argument('--fa_zs', type=str) # falign.score
    parser.add_argument('--pron_risk', type=str) # pron.score
    parser.add_argument('--pron_ref', type=str) # pron.score
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--ref_lid', type=str)
    parser.add_argument('--topk_lid', type=str)
    parser.add_argument('--ref_asr', type=str)
    parser.add_argument('--topk_asr', type=str)
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--s_scale', type=int, default = 10)
    parser.add_argument('--w_scale', type=int, default = 10)
    parser.add_argument('--a_scale', type=int, default = 1)
    parser.add_argument('--l_scale', type=int, default = 1)
    parser.add_argument('--fm_scale', type=int, default = 1)
    parser.add_argument('--fz_scale', type=int, default = 1)
    parser.add_argument('--prisk_scale', type=int, default = 1)
    parser.add_argument('--pref_scale', type=int, default = 1)
    parser.add_argument('--lm_norm', type=int, default = 0)
    parser.add_argument('--asr_length', type=str, default = None)
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    parser.add_argument('--length_score', type=int, default = 0)
    parser.add_argument('--temp', type=float, default = 1)
    parser.add_argument('--mean_asr', type=str, default = None)
    parser.add_argument('--mean_lm', type=str, default = None)
    args = parser.parse_args()

    slid = [x.strip() for x in open(args.slid, "r").readlines()]
    wlid = [x.strip() for x in open(args.wlid, "r").readlines()]
    asr = [x.strip() for x in open(args.asr, "r").readlines()]
    if args.asr_length is not None:
        asr_length = [x.strip() for x in open(args.asr_length, "r").readlines()]
    lm = [x.strip() for x in open(args.lm, "r").readlines()]
    fa_mms = [x.strip() for x in open(args.fa_mms, "r").readlines()]
    fa_zs = [x.strip() for x in open(args.fa_zs, "r").readlines()]
    pron_risk = [x.strip() for x in open(args.pron_risk, "r").readlines()]
    pron_ref = [x.strip() for x in open(args.pron_ref, "r").readlines()]
    if args.mean_asr is not None:
        mean_asr = [float(x.strip()) for x in open(args.mean_asr, "r").readlines()]
    if args.mean_lm is not None:
        mean_lm = [float(x.strip()) for x in open(args.mean_lm, "r").readlines()]

    assert len(slid) * args.k == len(wlid)
    assert len(wlid) == len(asr)
    assert len(asr) == len(lm)
    # assert len(pron) == len(asr)

    ref_lid = [x.strip() for x in open(args.ref_lid, "r").readlines()]
    topk_lid = [x.strip() for x in open(args.topk_lid, "r").readlines()]
    ref_asr = [x.strip() for x in open(args.ref_asr, "r").readlines()]
    topk_asr = [x.strip() for x in open(args.topk_asr, "r").readlines()]

    assert len(ref_lid) * args.k == len(topk_lid)
    assert len(ref_asr) * args.k == len(topk_asr)
    assert len(ref_lid) == len(ref_asr)

    slid_new = []
    for x in slid:
        data = eval(x)
        if args.temp == 1:
            for y in data:
                slid_new.append(math.log(y[1]))
        else:
            # import pdb;pdb.set_trace()
            new_data = torch.log_softmax(torch.tensor([x[1] for x in data]) / args.temp, dim=-1).tolist()
            for y in new_data:
                slid_new.append(y)


    wlid_new = []
    for x in wlid:
        data = eval(x)
        if data == 0:
            # wlid_new.append(math.log(0.000000001))
            wlid_new.append(-1000)
        else:
            # some values appear to exceed 1; need to look into fasttext norm
            wlid_new.append(math.log(data))

    asr_new = []
    for i, x in enumerate(asr):
        if x == "":
            asr_new.append(-1000)
        else:
            if args.asr_length is not None and args.length_score == 0:
            # if args.asr_length is not None:
                val = float(x) / int(asr_length[i])
            else:
                val = float(x)
            if args.mean_asr is not None:
                val -= mean_asr[i]
            asr_new.append(val)

    lm_new = []
    for x in lm:
        score, length = x.split("\t", 1)
        score = float(score)
        length = int(length)
        if args.lm_norm != 0:
            if length == 0:
                score = -1000
            else:
                score = score / length
        if args.mean_lm is not None:
                score -= mean_lm[i]
        lm_new.append(score)

    fa_mms_new = []
    for x in fa_mms:
        if x == "":
            fa_mms_new.append(-10000)
        elif x == "-inf":
            fa_mms_new.append(-10000)
        else:
            score = float(x)
            fa_mms_new.append(score)

    fa_zs_new = []
    for x in fa_zs:
        if x == "":
            fa_zs_new.append(-10000)
        else:
            score = float(x)
            fa_zs_new.append(score)

    pron_risk_new = []
    for x in pron_risk:
        score = float(x)
        pron_risk_new.append(math.log(score))

    pron_ref_new = []
    for x in pron_ref:
        score = float(x)
        pron_ref_new.append(math.log(score))

    if args.length_score == 1:
        len_new = []
        if args.asr_length is not None:
            for x in asr_length:
                len_new.append(int(x))
        else:
            for x in topk_asr:
                len_new.append(len(x))

    slid_new = normalize(slid_new)
    wlid_new = normalize(wlid_new)
    asr_new = normalize(asr_new)
    lm_new = normalize(lm_new)
    fa_mms_new = normalize(fa_mms_new)
    fa_zs_new = normalize(fa_zs_new)
    pron_risk_new = normalize(pron_risk_new)
    pron_ref_new = normalize(pron_ref_new)

    feats = [[s, w, a, l, fm, fz, prisk, pref, le] for s,w,a,l,fm,fz,prisk,pref,le in zip(slid_new, wlid_new, asr_new, lm_new, fa_mms_new, fa_zs_new, pron_risk_new, pron_ref_new, len_new)]

    # import pdb;pdb.set_trace()
    right = defaultdict(list)
    wrong = defaultdict(list)
    agg_right = []
    agg_wrong = []
    for i in range(len(topk_lid)):
        idx = i // args.k
        # skip empties
        if asr[i] == "":
            continue
        if topk_lid[i] == ref_lid[idx]:
            right[topk_lid[i]].append(feats[i])
            agg_right.append(feats[i])
        else:
            wrong[topk_lid[i]].append(feats[i])
            agg_wrong.append(feats[i])
    
    # plot(right, wrong)

    # import pdb;pdb.set_trace()
    langs = ["eng", "afr", "nld", "spa", "deu"]
    if not os.path.exists(args.dst+"/plots"):
        os.makedirs(args.dst+"/plots")
    for lang in langs:
        for i in range(len(feats[0])):
            plot([x[i] for x in right[lang]], [x[i] for x in wrong[lang]], args.dst + "/plots/"+lang+str(i)+".png")

    for i in range(len(feats[0])):
        print(i, np.array([x[i] for x in agg_right]).mean(), np.array([x[i] for x in agg_wrong]).mean())
        plot([x[i] for x in agg_right], [x[i] for x in agg_wrong], args.dst + "/plots/all"+str(i)+".png")