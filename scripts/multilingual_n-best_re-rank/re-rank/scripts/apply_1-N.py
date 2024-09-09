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

def select(w, feats, ref_lid, topk_lid, ref_asr, topk_asr, k=10, exclude=None):
    if len(w) == 4:
        scores = [w[0]*f[0] + w[1]*f[1] + w[2]*f[2] + w[3]*f[3] for f in feats]
    elif len(w) == 5:
        scores = [w[0]*f[0] + w[1]*f[1] + w[2]*f[2] + w[3]*f[3] + w[4]*f[4] for f in feats]

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

    og_asr_err = 0
    # og_asr_total = 0
    corr_og_asr_err = 0
    # corr_og_asr_total = 0
    err_og_asr_err = 0
    # err_og_asr_total = 0

    feat_vals_sel = [[] for _ in range(len(w))]
    feat_vals_not_sel = [[] for _ in range(len(w))]
    feat_vals_og = [[] for _ in range(len(w))]
    feat_vals_ora = [[] for _ in range(len(w))]

    ora_lid_correct = 0
    ora_asr_err = 0
    corr_ora_asr_err = 0
    err_ora_asr_err = 0
    corr_ora_lid_correct = 0
    err_ora_lid_correct = 0

    for i in range(len(ref_lid)):
        if exclude is not None:
            if ref_lid[i] in exclude:
                continue

        start_idx = i * 10
        end_idx = start_idx + k
        cand_scores = scores[start_idx:end_idx]
        max_idx, max_val = max(enumerate(cand_scores), key=lambda x: x[1])

        cand_feats = feats[start_idx:end_idx]
        for j in range(len(w)):
            feat_vals_og[j].append(cand_feats[0][j])
        for c in range(k):
            if c != max_idx:
                for j in range(len(w)):
                    feat_vals_not_sel[j].append(cand_feats[c][j])
            else:
                for j in range(len(w)):
                    feat_vals_sel[j].append(cand_feats[c][j])

        lang.append(topk_lid[start_idx:end_idx][max_idx])
        if ref_lid[i] == topk_lid[start_idx:end_idx][max_idx]:
            lid_correct += 1
        lid_total += 1

        hyp = topk_asr[start_idx:end_idx][max_idx]
        text.append(hyp)
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

        hyp_og = topk_asr[start_idx]
        hyp_og = hyp_og.lower()
        hyp_og = hyp_og.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
        if ref_lid[i] in cer_langs:
            hyp_og = " ".join(hyp_og)
        hyp_og_words = hyp_og.split()
        og_errs = editdistance.eval(hyp_og_words, tgt_words)
        og_asr_err += og_errs

        
        # lid-oracle
        cands_lid = topk_lid[start_idx:end_idx]
        try:
            ora_idx = cands_lid.index(ref_lid[i])
            ora_lid_correct += 1
        except:
            ora_idx = 0

        hyp_ora = topk_asr[start_idx:end_idx][ora_idx]
        hyp_ora = hyp_ora.lower()
        hyp_ora = hyp_ora.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
        if ref_lid[i] in cer_langs:
            hyp_ora = " ".join(hyp_ora)
        hyp_ora_words = hyp_ora.split()
        ora_errs = editdistance.eval(hyp_ora_words, tgt_words)
        ora_asr_err += ora_errs

        for j in range(len(w)):
            feat_vals_ora[j].append(cand_feats[ora_idx][j])

        if ref_lid[i] == topk_lid[start_idx]:
            if topk_lid[start_idx:end_idx][max_idx] == ref_lid[i]:
                corr_lid_correct += 1
            corr_lid_total += 1
            corr_asr_err += errs
            corr_asr_total += len(tgt_words)

            corr_og_asr_err += og_errs

            corr_ora_asr_err += ora_errs
            if cands_lid[ora_idx] == ref_lid[i]:
                corr_ora_lid_correct += 1
        else:
            if topk_lid[start_idx:end_idx][max_idx] == ref_lid[i]:
                err_lid_correct += 1
            err_lid_total += 1
            err_asr_err += errs
            err_asr_total += len(tgt_words)

            err_og_asr_err += og_errs

            err_ora_asr_err += ora_errs
            if cands_lid[ora_idx] == ref_lid[i]:
                err_ora_lid_correct += 1

    results = {"lid_acc": lid_correct / lid_total, "asr_wer": asr_err / asr_total, "weights": w}
    results["corr_lid_acc"] = corr_lid_correct / corr_lid_total
    results["err_lid_acc"] = err_lid_correct / err_lid_total
    results["corr_asr_wer"] = corr_asr_err / corr_asr_total
    results["err_asr_wer"] = err_asr_err / err_asr_total
    
    results["og_asr_wer"] = og_asr_err / asr_total
    results["og_corr_asr_wer"] = corr_og_asr_err / corr_asr_total
    results["og_err_asr_wer"] = err_og_asr_err / err_asr_total

    results["ora_asr_wer"] = ora_asr_err / asr_total
    results["ora_corr_asr_wer"] = corr_ora_asr_err / corr_asr_total
    results["ora_err_asr_wer"] = err_ora_asr_err / err_asr_total
    results["ora_lid_acc"] = ora_lid_correct / lid_total
    results["ora_corr_lid_acc"] = corr_ora_lid_correct / corr_lid_total
    results["ora_err_lid_acc"] = err_ora_lid_correct / err_lid_total

    feat_vals_sel = [np.array(x).mean() * w[i] for i, x in enumerate(feat_vals_sel)]
    results["feat_vals_sel"] = feat_vals_sel

    feat_vals_not_sel = [np.array(x).mean() * w[i] for i, x in enumerate(feat_vals_not_sel)]
    results["feat_vals_not_sel"] = feat_vals_not_sel

    feat_vals_og = [np.array(x).mean() * w[i] for i, x in enumerate(feat_vals_og)]
    results["feat_vals_og"] = feat_vals_og

    feat_vals_ora = [np.array(x).mean() * w[i] for i, x in enumerate(feat_vals_ora)]
    results["feat_vals_ora"] = feat_vals_ora

    return results, text, lang

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--slid', type=str) # predictions.txt
    parser.add_argument('--wlid', type=str) # predictions.txt.scores
    parser.add_argument('--asr', type=str)  # hypo.score
    parser.add_argument('--lm', type=str)  # predictions.txt
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
    parser.add_argument('--w', type=str)
    parser.add_argument('--tag', type=str, default = None)
    parser.add_argument('--length_score', type=int, default = 0)
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    args = parser.parse_args()

    if args.length_score == 1:
        s_w, w_w, a_w, l_w, le_w = eval(open(args.w, "r").read())['weights']
    else:
        s_w, w_w, a_w, l_w = eval(open(args.w, "r").read())['weights']


    slid = [x.strip() for x in open(args.slid, "r").readlines()]
    wlid = [x.strip() for x in open(args.wlid, "r").readlines()]
    asr = [x.strip() for x in open(args.asr, "r").readlines()]
    lm = [x.strip() for x in open(args.lm, "r").readlines()]
    if args.asr_length is not None:
        asr_length = [x.strip() for x in open(args.asr_length, "r").readlines()]
        assert len(asr_length) == len(asr)

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
            slid_new.append(math.log(y[1]))

    wlid_new = []
    for x in wlid:
        data = eval(x)
        if data == 0:
            wlid_new.append(math.log(0.000000001))
        else:
            # some values appear to exceed 1; need to look into fasttext norm
            wlid_new.append(math.log(data))

    asr_new = []
    for x in asr:
        if x == "":
            asr_new.append(-1000)
        else:
            asr_new.append(float(x))

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
    
    if args.length_score == 1:
        len_new = []
        if args.asr_length is not None:
            for x in asr_length:
                len_new.append(int(x))
        else:
            for x in topk_asr:
                len_new.append(len(x))

    if args.norm != 0:
        # import pdb;pdb.set_trace()
        slid_new = normalize(slid_new)
        wlid_new = normalize(wlid_new)
        asr_new = normalize(asr_new)
        lm_new = normalize(lm_new)
        s_scale = 1
        w_scale = args.w_scale
    else:
        s_scale = 10
        w_scale = 10

    if args.length_score == 1:
        feats = [[s, w, a, l, le] for s,w,a,l,le in zip(slid_new, wlid_new, asr_new, lm_new, len_new)]
    else:
        feats = [[s, w, a, l] for s,w,a,l in zip(slid_new, wlid_new, asr_new, lm_new)]

    
    
    
    # s_w = s_w * s_scale
    # w_w = w_w * w_scale
    # a_w = a_w
    # l_w = l_w
    if args.length_score == 1:
        weight = [s_w, w_w, a_w, l_w, le_w]
    else:
        weight = [s_w, w_w, a_w, l_w]

    # import pdb;pdb.set_trace()
    for i in range(args.k):
        results, text, lang = select(weight, feats, ref_lid, topk_lid, ref_asr, topk_asr, k=i+1, exclude=args.exclude)

        if args.tag is not None:
            tag_text = "." + args.tag
        else:
            tag_text = ""

        tag_text += ".k="+str(i+1)

        with open(args.dst + "/text" + tag_text, "w") as f_out:
            f_out.writelines([x+"\n" for x in text])

        with open(args.dst + "/lang" + tag_text, "w") as f_out:
            f_out.writelines([x+"\n" for x in lang])

        with open(args.dst + "/text.result" + tag_text, "w") as f_out:
            # f_out.write(str(results))
            for k in results.keys():
                f_out.write(k + "\t" + str(results[k]) + "\n")
