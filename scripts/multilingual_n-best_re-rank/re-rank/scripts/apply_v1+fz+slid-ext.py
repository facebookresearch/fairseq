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
    # if len(w) == 4:
    #     scores = [w[0]*f[0] + w[1]*f[1] + w[2]*f[2] + w[3]*f[3] for f in feats]
    # elif len(w) == 5:
    #     scores = [w[0]*f[0] + w[1]*f[1] + w[2]*f[2] + w[3]*f[3] + w[4]*f[4] for f in feats]

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

    og_lid_correct = 0
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

    # lid oracle
    ora_lid_correct = 0
    ora_asr_err = 0
    corr_ora_asr_err = 0
    err_ora_asr_err = 0
    corr_ora_lid_correct = 0
    err_ora_lid_correct = 0

    # wer oracle
    ora2_lid_correct = 0
    ora2_asr_err = 0
    corr_ora2_asr_err = 0
    err_ora2_asr_err = 0
    corr_ora2_lid_correct = 0
    err_ora2_lid_correct = 0

    for i in range(len(ref_lid)):
        if exclude is not None:
            if ref_lid[i] in exclude:
                continue

        start_idx = i * k
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
        if ref_lid[i] == topk_lid[start_idx]:
            og_lid_correct += 1

        
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

        # wer-oracle
        cand_wers = []
        for h in topk_asr[start_idx:end_idx]:
            h = h.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
            if ref_lid[i] in cer_langs:
                h = " ".join(h)
            h_words = h.split()
            h_errs = editdistance.eval(h_words, tgt_words)
            cand_wers.append(h_errs)
        ora2_idx, ora2_val = min(enumerate(cand_wers), key=lambda x: x[1])
        ora2_asr_err += ora2_val
        if cands_lid[ora2_idx] == ref_lid[i]:
            ora2_lid_correct += 1

        # if cands_lid[0] != cands_lid[max_idx]:
        #     print("=================================")
        #     if cands_lid[ora_idx] == cands_lid[max_idx]:
        #         print("TYPE: CORRECT")
        #     else:
        #         print("TYPE: INCORRECT")
        #     if ref_lid[i] == topk_lid[start_idx]:
        #         print("SUBSET: ORIGINALLY CORRECT LID")
        #     else:
        #         print("SUBSET: ORIGINALLY INCORRECT LID")

        #     feat_names = ["SPOKEN_LID_SCORE", "WRITTEN_LID_SCORE", "ASR_CONF_SCORE", "LM_CONF_SCORE", "FALIGN_MMS", "FALIGN_ZS", "UROMAN-ERR-RATE_RISK", "UROMAN-ERR-RATE_REF", "LENGTH_SCORE"]
        #     rescore_vs_base = [abs(cand_feats[0][i]*w[i] - cand_feats[max_idx][i]*w[i]) for i in range(len(w))]
        #     rescore_vs_ora = [abs(cand_feats[ora_idx][i]*w[i] - cand_feats[max_idx][i]*w[i]) for i in range(len(w))]
        #     print("re-score vs baseline is most influenced by:", feat_names[rescore_vs_base.index(max(rescore_vs_base))], round(max(rescore_vs_base), 2))
        #     print("re-score vs oracle is most influenced by:", feat_names[rescore_vs_ora.index(max(rescore_vs_ora))], round(max(rescore_vs_ora), 2))
        #     print("")

        #     print("------BASELINE------")
        #     print("LID:", cands_lid[0])
        #     print("HYP:", topk_asr[start_idx:end_idx][0])
        #     print("SPOKEN_LID_SCORE:", round(cand_feats[0][0], 2), "*", round(w[0], 2), "=", round(cand_feats[0][0]*w[0], 2))
        #     print("WRITTEN_LID_SCORE:", round(cand_feats[0][1], 2), "*", round(w[1], 2), "=", round(cand_feats[0][1]*w[1], 2))
        #     print("ASR_CONF_SCORE:", round(cand_feats[0][2], 2), "*", round(w[2], 2), "=", round(cand_feats[0][2]*w[2], 2))
        #     print("LM_CONF_SCORE:", round(cand_feats[0][3], 2), "*", round(w[3], 2), "=", round(cand_feats[0][3]*w[3], 2))
        #     print("FALIGN_MMS:", round(cand_feats[0][4], 2), "*", round(w[4], 2), "=", round(cand_feats[0][4]*w[4], 2))
        #     print("FALIGN_ZS:", round(cand_feats[0][5], 2), "*", round(w[5], 2), "=", round(cand_feats[0][5]*w[5], 2))
        #     print("UROMAN-ERR-RATE_RISK:", round(cand_feats[0][6], 2), "*", round(w[6], 2), "=", round(cand_feats[0][6]*w[6], 2))
        #     print("UROMAN-ERR-RATE_REF:", round(cand_feats[0][7], 2), "*", round(w[7], 2), "=", round(cand_feats[0][7]*w[7], 2))
        #     print("LENGTH_SCORE:", round(cand_feats[0][8], 2), "*", round(w[8], 2), "=", round(cand_feats[0][8]*w[8], 2))
        #     print("TOTAL_SCORE:", round(cand_scores[0], 2))
        #     print("")

        #     print("------ORACLE------")
        #     print("LID:", cands_lid[ora_idx])
        #     print("HYP:", topk_asr[start_idx:end_idx][ora_idx])
        #     print("SPOKEN_LID_SCORE:", round(cand_feats[ora_idx][0], 2), "*", round(w[0], 2), "=", round(cand_feats[ora_idx][0]*w[0], 2))
        #     print("WRITTEN_LID_SCORE:", round(cand_feats[ora_idx][1], 2), "*", round(w[1], 2), "=", round(cand_feats[ora_idx][1]*w[1], 2))
        #     print("ASR_CONF_SCORE:", round(cand_feats[ora_idx][2], 2), "*", round(w[2], 2), "=", round(cand_feats[ora_idx][2]*w[2], 2))
        #     print("LM_CONF_SCORE:", round(cand_feats[ora_idx][3], 2), "*", round(w[3], 2), "=", round(cand_feats[ora_idx][3]*w[3], 2))
        #     print("FALIGN_MMS:", round(cand_feats[ora_idx][4], 2), "*", round(w[4], 2), "=", round(cand_feats[ora_idx][4]*w[4], 2))
        #     print("FALIGN_ZS:", round(cand_feats[ora_idx][5], 2), "*", round(w[5], 2), "=", round(cand_feats[ora_idx][5]*w[5], 2))
        #     print("UROMAN-ERR-RATE_RISK:", round(cand_feats[ora_idx][6], 2), "*", round(w[6], 2), "=", round(cand_feats[ora_idx][6]*w[6], 2))
        #     print("UROMAN-ERR-RATE_REF:", round(cand_feats[ora_idx][7], 2), "*", round(w[7], 2), "=", round(cand_feats[ora_idx][7]*w[7], 2))
        #     print("LENGTH_SCORE:", round(cand_feats[ora_idx][8], 2), "*", round(w[8], 2), "=", round(cand_feats[ora_idx][8]*w[8], 2))
        #     print("TOTAL_SCORE:", round(cand_scores[ora_idx], 2))
        #     print("")
        #     # print("O", cands_lid[ora_idx], topk_asr[start_idx:end_idx][ora_idx], cand_feats[ora_idx])

        #     print("------RE-SCORE------")
        #     print("LID:", cands_lid[max_idx])
        #     print("HYP:", topk_asr[start_idx:end_idx][max_idx])
        #     print("SPOKEN_LID_SCORE:", round(cand_feats[max_idx][0], 2), "*", round(w[0], 2), "=", round(cand_feats[max_idx][0]*w[0], 2))
        #     print("WRITTEN_LID_SCORE:", round(cand_feats[max_idx][1], 2), "*", round(w[1], 2), "=", round(cand_feats[max_idx][1]*w[1], 2))
        #     print("ASR_CONF_SCORE:", round(cand_feats[max_idx][2], 2), "*", round(w[2], 2), "=", round(cand_feats[max_idx][2]*w[2], 2))
        #     print("LM_CONF_SCORE:", round(cand_feats[max_idx][3], 2), "*", round(w[3], 2), "=", round(cand_feats[max_idx][3]*w[3], 2))
        #     print("FALIGN_MMS:", round(cand_feats[max_idx][4], 2), "*", round(w[4], 2), "=", round(cand_feats[max_idx][4]*w[4], 2))
        #     print("FALIGN_ZS:", round(cand_feats[max_idx][5], 2), "*", round(w[5], 2), "=", round(cand_feats[max_idx][5]*w[5], 2))
        #     print("UROMAN-ERR-RATE_RISK:", round(cand_feats[max_idx][6], 2), "*", round(w[6], 2), "=", round(cand_feats[max_idx][6]*w[6], 2))
        #     print("UROMAN-ERR-RATE_REF:", round(cand_feats[max_idx][7], 2), "*", round(w[7], 2), "=", round(cand_feats[max_idx][7]*w[7], 2))
        #     print("LENGTH_SCORE:", round(cand_feats[max_idx][8], 2), "*", round(w[8], 2), "=", round(cand_feats[max_idx][8]*w[8], 2))
        #     print("TOTAL_SCORE:", round(cand_scores[max_idx], 2))
        #     print("")

        for j in range(len(w)):
            feat_vals_ora[j].append(cand_feats[ora_idx][j])

        # corr
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

            corr_ora2_asr_err += ora2_val
            if cands_lid[ora2_idx] == ref_lid[i]:
                corr_ora2_lid_correct += 1
        # err
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

            err_ora2_asr_err += ora2_val
            if cands_lid[ora2_idx] == ref_lid[i]:
                err_ora2_lid_correct += 1

    results = {"lid_acc": lid_correct / lid_total, "asr_wer": asr_err / asr_total, "weights": w}
    results["corr_lid_acc"] = corr_lid_correct / corr_lid_total
    results["err_lid_acc"] = err_lid_correct / err_lid_total
    results["corr_asr_wer"] = corr_asr_err / corr_asr_total
    results["err_asr_wer"] = err_asr_err / err_asr_total
    
    results["og_lid_acc"] = og_lid_correct / lid_total
    results["og_asr_wer"] = og_asr_err / asr_total
    results["og_corr_asr_wer"] = corr_og_asr_err / corr_asr_total
    results["og_err_asr_wer"] = err_og_asr_err / err_asr_total

    results["ora_asr_wer"] = ora_asr_err / asr_total
    results["ora_corr_asr_wer"] = corr_ora_asr_err / corr_asr_total
    results["ora_err_asr_wer"] = err_ora_asr_err / err_asr_total
    results["ora_lid_acc"] = ora_lid_correct / lid_total
    results["ora_corr_lid_acc"] = corr_ora_lid_correct / corr_lid_total
    results["ora_err_lid_acc"] = err_ora_lid_correct / err_lid_total

    results["ora2_asr_wer"] = ora2_asr_err / asr_total
    results["ora2_corr_asr_wer"] = corr_ora2_asr_err / corr_asr_total
    results["ora2_err_asr_wer"] = err_ora2_asr_err / err_asr_total
    results["ora2_lid_acc"] = ora2_lid_correct / lid_total
    results["ora2_corr_lid_acc"] = corr_ora2_lid_correct / corr_lid_total
    results["ora2_err_lid_acc"] = err_ora2_lid_correct / err_lid_total

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
    parser.add_argument('--w', type=str)
    parser.add_argument('--tag', type=str, default = None)
    parser.add_argument('--length_score', type=int, default = 0)
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    parser.add_argument('--mean_conf', type=str, default = None)
    args = parser.parse_args()

    if args.length_score == 1:
        s_w, se_w, w_w, a_w, l_w, fz_w, le_w = eval(open(args.w, "r").read())['weights']
    else:
        s_w, se_w, w_w, a_w, l_w, fz_w = eval(open(args.w, "r").read())['weights']

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
            slid_new.append(math.log(y[1]))

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
        feats = [[s, se, w, a, l, fz, le] for s,se,w,a,l,fz,le in zip(slid_new, slid_ext_new, wlid_new, asr_new, lm_new, fa_zs_new, len_new)]
    else:
        feats = [[s, se, w, a, l, fz] for s,se,w,a,l,fz in zip(slid_new, slid_ext_new, wlid_new, asr_new, lm_new, fa_zs_new)]
    
    if args.length_score == 1:
        weight = [s_w, se_w, w_w, a_w, l_w, fz_w, le_w]
    else:
        weight = [s_w, se_w, w_w, a_w, l_w, fz_w]

    # import pdb;pdb.set_trace()
    results, text, lang = select(weight, feats, ref_lid, topk_lid, ref_asr, topk_asr, k=args.k, exclude=args.exclude)

    if args.tag is not None:
        tag_text = "." + args.tag
    else:
        tag_text = ""

    with open(args.dst + "/text" + tag_text, "w") as f_out:
        f_out.writelines([x+"\n" for x in text])

    with open(args.dst + "/lang" + tag_text, "w") as f_out:
        f_out.writelines([x+"\n" for x in lang])

    with open(args.dst + "/text.result" + tag_text, "w") as f_out:
        # f_out.write(str(results))
        for k in results.keys():
            f_out.write(k + "\t" + str(results[k]) + "\n")
