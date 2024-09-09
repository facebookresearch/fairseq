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

def compute(w, feats, ref_lid, topk_lid, ref_asr, topk_asr, k=10, exclude=None):
    assert len(w) == len(feats[0])
    scores = []
    for f in feats:
        s = 0
        for i in range(len(w)):
            s += w[i]*f[i]
        scores.append(s)

    lid_correct = 0
    lid_total = 0
    asr_err = 0
    asr_total = 0

    for i in range(len(ref_lid)):
        if exclude is not None:
            if ref_lid[i] in exclude:
                continue

        start_idx = i * k
        end_idx = start_idx + k
        cand_scores = scores[start_idx:end_idx]
        max_idx, max_val = max(enumerate(cand_scores), key=lambda x: x[1])

        if ref_lid[i] == topk_lid[start_idx:end_idx][max_idx]:
            lid_correct += 1
        lid_total += 1


        hyp = topk_asr[start_idx:end_idx][max_idx]
        ref = ref_asr[i]
        hyp = hyp.lower()
        ref = ref.lower()
        hyp = hyp.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
        ref = ref.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
        if ref_lid[i] in cer_langs:
            hyp = " ".join(hyp)
            ref = " ".join(ref)

        hyp_words = hyp.split()
        tgt_words = ref.split()
        errs = editdistance.eval(hyp_words, tgt_words)
        asr_err += errs
        asr_total += len(tgt_words)

    return {"lid_acc": lid_correct / lid_total, "asr_wer": asr_err / asr_total, "weights": w}

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
    parser.add_argument('--oracle_flag', type=str, default = None)
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
    oracle_flag = [x.strip() for x in open(args.oracle_flag, "r").readlines()]

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
    
    oracle_flag_new = []
    for x in oracle_flag:
        score = int(x)
        oracle_flag_new.append(score)

    if args.norm != 0:
        # import pdb;pdb.set_trace()
        slid_new = normalize(slid_new)
        wlid_new = normalize(wlid_new)
        asr_new = normalize(asr_new)
        lm_new = normalize(lm_new)
        fa_mms_new = normalize(fa_mms_new)
        fa_zs_new = normalize(fa_zs_new)
        pron_risk_new = normalize(pron_risk_new)
        pron_ref_new = normalize(pron_ref_new)
        oracle_flag_new = normalize(oracle_flag_new)
        s_scale = 1 if args.s_scale != 0 else 0
        w_scale = 1 if args.w_scale != 0 else 0
        a_scale = 1 if args.a_scale != 0 else 0
        l_scale = 1 if args.l_scale != 0 else 0
        fm_scale = 1 if args.fm_scale != 0 else 0
        fz_scale = 1 if args.fz_scale != 0 else 0
        prisk_scale = 1 if args.prisk_scale != 0 else 0
        pref_scale = 1 if args.pref_scale != 0 else 0
    else:
        s_scale = args.s_scale
        w_scale = args.w_scale
        a_scale = args.a_scale
        l_scale = args.l_scale
        fm_scale = args.fm_scale
        fz_scale = args.fz_scale
        prisk_scale = args.prisk_scale
        pref_scale = args.pref_scale

    if args.length_score == 1:
        feats = [[s, w, a, l, fm, fz, prisk, pref, o, le] for s,w,a,l,fm,fz,prisk,pref,o,le in zip(slid_new, wlid_new, asr_new, lm_new, fa_mms_new, fa_zs_new, pron_risk_new, pron_ref_new, oracle_flag_new, len_new)]
    else:
        feats = [[s, w, a, l, fm, fz, prisk, pref, o] for s,w,a,l,fm,fz,prisk,pref,o in zip(slid_new, wlid_new, asr_new, lm_new, fa_mms_new, fa_zs_new, pron_risk_new, pron_ref_new, oracle_flag_new)]
    
    weights = []
    for i in range(args.iters):
        s_w = np.random.rand() * s_scale
        w_w = np.random.rand() * w_scale
        a_w = np.random.rand() * a_scale
        l_w = np.random.rand() * l_scale
        fm_w = np.random.rand() * fm_scale
        fz_w = np.random.rand() * fz_scale
        prisk_w = -np.random.rand() * prisk_scale
        pref_w = -np.random.rand() * pref_scale
        o_w = np.random.rand()
        if args.length_score == 1:
            le_w = (np.random.rand() -0.5) * 1
            weights.append([s_w, w_w, a_w, l_w, fm_w, fz_w, prisk_w, pref_w, o_w, le_w])
        else:
            weights.append([s_w, w_w, a_w, l_w, fm_w, fz_w, prisk_w, pref_w, o_w])

    # import pdb;pdb.set_trace()
    num_tries = len(weights)
    print("Total number of search points", num_tries)
    threads = 64
    pool = Pool(threads)
    compute_fxn = partial(compute, feats=feats, ref_lid=ref_lid, topk_lid=topk_lid, ref_asr=ref_asr, topk_asr=topk_asr, k=args.k, exclude=args.exclude)
    results = pool.map(compute_fxn, weights)
    pool.close()
    pool.join()

    assert len(results) == len(weights)

    lid_best = 0
    wer_best = 100
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)
    with open(args.dst + "/results.all", "w") as f_out:
        for result in results:
            f_out.write(str(result)+"\n")
            if result["lid_acc"] > lid_best:
                lid_best = result["lid_acc"]
            if result["asr_wer"] < wer_best:
                wer_best = result["asr_wer"]


    with open(args.dst + "/results.best", "w") as f_out:
        f_out.write("BEST LID ACC " + str(lid_best) + "\n")
        f_out.write("BEST ASR WER " + str(wer_best) + "\n")
