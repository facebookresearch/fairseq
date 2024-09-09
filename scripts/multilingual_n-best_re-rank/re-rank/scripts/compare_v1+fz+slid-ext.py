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

cer_langs = [x.strip() for x in open("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/cer_langs.txt", "r").readlines()]

def normalize(feat):
    # create a StandardScaler object
    scaler = StandardScaler()
    # fit the scaler to the data
    X = np.array(feat).reshape(-1,1)
    scaler.fit(X)
    # transform the data
    X_normalized = scaler.transform(X)
    return X_normalized.flatten().tolist()

def select(w1, w2, feats1, feats2, ref_lid, topk_lid, ref_asr, topk_asr, k=10, exclude=None):
    assert len(w1) == len(feats1[0])
    scores1 = []
    for f in feats1:
        s = 0
        for i in range(len(w1)):
            s += w1[i]*f[i]
        scores1.append(s)

    assert len(w2) == len(feats2[0])
    scores2 = []
    for f in feats2:
        s = 0
        for i in range(len(w2)):
            s += w2[i]*f[i]
        scores2.append(s)

    lid_correct = 0
    lid_total = 0
    asr_err = 0
    asr_total = 0
    text = []
    lang = []

    corr_lid_correct = 0
    corr_lid_total = 0
    corr_asr_err = 0
    corr_asr_total = 0

    err_lid_correct = 0
    err_lid_total = 0
    err_asr_err = 0
    err_asr_total = 0

    count = 0
    pairs = defaultdict(int)
    corr_langs = defaultdict(int)
    wrong_langs = defaultdict(int)

    for i in range(len(ref_lid)):
        if exclude is not None:
            if ref_lid[i] in exclude:
                continue

        start_idx = i * k
        end_idx = start_idx + k

        cand_scores1 = scores1[start_idx:end_idx]
        max_idx1, max_val1 = max(enumerate(cand_scores1), key=lambda x: x[1])

        cand_scores2 = scores2[start_idx:end_idx]
        max_idx2, max_val2 = max(enumerate(cand_scores2), key=lambda x: x[1])
        
        if max_idx1 != max_idx2 and topk_lid[start_idx:end_idx][max_idx1] == ref_lid[i]:
            count += 1
            # import pdb;pdb.set_trace()
            print("Ref:", topk_lid[start_idx:end_idx][max_idx1], "Hyp:", topk_lid[start_idx:end_idx][max_idx2])
            # print("Ref_Whisper-LID-feat:", math.exp(feats1[start_idx:end_idx][max_idx1][0]), "Ref_MMS-LID-feat:", math.exp(feats1[start_idx:end_idx][max_idx1][1]))
            # print("Hyp_Whisper-LID-feat:", math.exp(feats1[start_idx:end_idx][max_idx2][0]), "Hyp_MMS-LID-feat:", math.exp(feats1[start_idx:end_idx][max_idx2][1]))
            print(math.exp(feats1[start_idx:end_idx][max_idx1][0]), math.exp(feats1[start_idx:end_idx][max_idx1][1]), feats1[start_idx:end_idx][max_idx1][2:])
            print(math.exp(feats1[start_idx:end_idx][max_idx2][0]), math.exp(feats1[start_idx:end_idx][max_idx2][1]), feats1[start_idx:end_idx][max_idx2][2:])
            print(max_idx1, topk_asr[start_idx:end_idx][max_idx1])
            print(max_idx2, topk_asr[start_idx:end_idx][max_idx2])
            print("---")
            pairs[topk_lid[start_idx:end_idx][max_idx1]+"-"+topk_lid[start_idx:end_idx][max_idx2]] += 1
            corr_langs[topk_lid[start_idx:end_idx][max_idx1]] += 1
            wrong_langs[topk_lid[start_idx:end_idx][max_idx2]] += 1

    print("Count", count)
    print("---")
    print("Confusion pair counts")
    print(sorted(pairs.items(), key=lambda x: x[1], reverse=True))
    print("---")
    print("Ref lang counts")
    print(sorted(corr_langs.items(), key=lambda x: x[1], reverse=True))
    print("---")
    print("Hyp lang counts")
    print(sorted(wrong_langs.items(), key=lambda x: x[1], reverse=True))


    # return results, text, lang

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--slid', type=str) # predictions.txt
    parser.add_argument('--slid_ext', type=str) # .mms_score
    parser.add_argument('--wlid', type=str) # predictions.txt.scores
    parser.add_argument('--asr', type=str)  # hypo.score
    parser.add_argument('--lm', type=str)  # predictions.txt
    parser.add_argument('--fa_zs', type=str) # falign.score
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--ref_lid', type=str)
    parser.add_argument('--topk_lid', type=str)
    parser.add_argument('--ref_asr', type=str)
    parser.add_argument('--topk_asr', type=str)
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--w_scale', type=int, default = 1)
    parser.add_argument('--lm_norm', type=int, default = 0)
    parser.add_argument('--asr_length', type=str, default = None)
    parser.add_argument('--w1', type=str)   # w slid_ext
    parser.add_argument('--w2', type=str)   # w/o slid_ext
    parser.add_argument('--tag', type=str, default = None)
    parser.add_argument('--length_score', type=int, default = 0)
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    parser.add_argument('--mean_conf', type=str, default = None)
    parser.add_argument('--clip', type=int, default = 0)
    parser.add_argument('--clip_thres', type=float, default = 0.01)
    parser.add_argument('--clip_conf', type=int, default = 0)
    parser.add_argument('--clip_conf_thres', type=float, default = -40)
    args = parser.parse_args()

    slid = [x.strip() for x in open(args.slid, "r").readlines()]
    slid_ext = [x.strip() for x in open(args.slid_ext, "r").readlines()]
    wlid = [x.strip() for x in open(args.wlid, "r").readlines()]
    asr = [x.strip() for x in open(args.asr, "r").readlines()]
    lm = [x.strip() for x in open(args.lm, "r").readlines()]
    if args.asr_length is not None:
        asr_length = [x.strip() for x in open(args.asr_length, "r").readlines()]
        assert len(asr_length) == len(asr)
    fa_zs = [x.strip() for x in open(args.fa_zs, "r").readlines()]
    if args.mean_conf is not None:
        mean_conf = [float(x.strip()) for x in open(args.mean_conf, "r").readlines()]

    assert len(slid) * args.k == len(wlid)
    assert len(wlid) == len(asr)
    assert len(asr) == len(lm)

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
        for y in data:
            if args.clip == 1 and y[1] < args.clip_thres:
                val = -1000
            else:
                val = math.log(y[1])
            slid_new.append(val)

    slid_ext_new = []
    for x in slid_ext:
        data = float(x)
        if data == 0:
            # wlid_new.append(math.log(0.000000001))
            slid_ext_new.append(-1000)
        else:
            # some values appear to exceed 1; need to look into fasttext norm
            slid_ext_new.append(math.log(data))

    wlid_new = []
    for x in wlid:
        data = eval(x)
        if data == 0:
            wlid_new.append(math.log(0.000000001))
        else:
            # some values appear to exceed 1; need to look into fasttext norm
            wlid_new.append(math.log(data))

    asr_new = []
    for i, x in enumerate(asr):
        if x == "":
            asr_new.append(-1000)
        else:
            val = float(x)
            if args.mean_conf is not None:
                if args.clip_conf == 1 and mean_conf[i] < args.clip_conf_thres:
                    val -= args.clip_conf_thres
                else:
                    val -= mean_conf[i]
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
        lm_new.append(score)

    fa_zs_new = []
    for x in fa_zs:
        if x == "":
            fa_zs_new.append(-10000)
        else:
            score = float(x)
            fa_zs_new.append(score)

    if args.length_score == 1:
        len_new = []
        if args.asr_length is not None:
            for x in asr_length:
                len_new.append(int(x))
        else:
            for x in topk_asr:
                len_new.append(len(x))

    if args.length_score == 1:
        feats1 = [[s, se, w, a, l, fz, le] for s,se,w,a,l,fz,le in zip(slid_new, slid_ext_new, wlid_new, asr_new, lm_new, fa_zs_new, len_new)]
        feats2 = [[s, w, a, l, fz, le] for s,w,a,l,fz,le in zip(slid_new, wlid_new, asr_new, lm_new, fa_zs_new, len_new)]

    else:
        feats1 = [[s, se, w, a, l, fz] for s,se,w,a,l,fz in zip(slid_new, slid_ext_new, wlid_new, asr_new, lm_new, fa_zs_new)]
        feats2 = [[s, w, a, l, fz] for s,w,a,l,fz in zip(slid_new, wlid_new, asr_new, lm_new, fa_zs_new)]
    
    if args.length_score == 1:
        s_w, se_w, w_w, a_w, l_w, fz_w, le_w = eval(open(args.w1, "r").read())['weights']
        weight1 = [s_w, se_w, w_w, a_w, l_w, fz_w, le_w]

        s_w, w_w, a_w, l_w, fz_w, le_w = eval(open(args.w2, "r").read())['weights']
        weight2 = [s_w, w_w, a_w, l_w, fz_w, le_w]
    else:
        s_w, se_w, w_w, a_w, l_w, fz_w = eval(open(args.w1, "r").read())['weights']
        weight1 = [s_w, se_w, w_w, a_w, l_w, fz_w]

        s_w, w_w, a_w, l_w, fz_w = eval(open(args.w2, "r").read())['weights']
        weight2 = [s_w, w_w, a_w, l_w, fz_w]


    # import pdb;pdb.set_trace()
    select(weight1, weight2, feats1, feats2, ref_lid, topk_lid, ref_asr, topk_asr, k=args.k, exclude=args.exclude)

    # if args.tag is not None:
    #     tag_text = "." + args.tag
    # else:
    #     tag_text = ""

    # with open(args.dst + "/text" + tag_text, "w") as f_out:
    #     f_out.writelines([x+"\n" for x in text])

    # with open(args.dst + "/lang" + tag_text, "w") as f_out:
    #     f_out.writelines([x+"\n" for x in lang])

    # with open(args.dst + "/text.result" + tag_text, "w") as f_out:
    #     # f_out.write(str(results))
    #     for k in results.keys():
    #         f_out.write(k + "\t" + str(results[k]) + "\n")
