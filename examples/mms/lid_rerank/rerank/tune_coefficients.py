import argparse
import os
from tqdm import tqdm
import numpy as np
import editdistance
from multiprocessing import Pool
from functools import partial

cer_langs = [x.strip() for x in open("cer_langs.txt", "r").readlines()]

def compute(w, feats, ref_lid, nbest_lid, ref_asr, nbest_asr, n=10, exclude=None):
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

        start_idx = i * n
        end_idx = start_idx + n
        cand_scores = scores[start_idx:end_idx]
        max_idx, max_val = max(enumerate(cand_scores), key=lambda x: x[1])

        if ref_lid[i] == nbest_lid[start_idx:end_idx][max_idx]:
            lid_correct += 1
        lid_total += 1

        hyp = nbest_asr[start_idx:end_idx][max_idx]
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
    parser.add_argument('--slid', type=str)
    parser.add_argument('--wlid', type=str)
    parser.add_argument('--asr', type=str)
    parser.add_argument('--lm', type=str)
    parser.add_argument('--uasr', type=str)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--ref_lid', type=str)
    parser.add_argument('--nbest_lid', type=str)
    parser.add_argument('--ref_asr', type=str)
    parser.add_argument('--nbest_asr', type=str)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--slid_scale', type=int, default = 100)
    parser.add_argument('--wlid_scale', type=int, default = 100)
    parser.add_argument('--asr_scale', type=int, default = 10)
    parser.add_argument('--lm_scale', type=int, default = 10)
    parser.add_argument('--uasr_scale', type=int, default = 10)
    parser.add_argument('--len_scale', type=int, default = 1)
    parser.add_argument('--num_jobs', type=int, default = 64)
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    args = parser.parse_args()

    slid = [float(x.strip()) for x in open(args.slid, "r").readlines()]
    wlid = [float(x.strip()) for x in open(args.wlid, "r").readlines()]
    asr = [float(x.strip()) for x in open(args.asr, "r").readlines()]
    lm = [float(x.strip()) for x in open(args.lm, "r").readlines()]
    uasr = [float(x.strip()) for x in open(args.uasr, "r").readlines()]

    assert len(slid) == len(wlid)
    assert len(wlid) == len(asr)
    assert len(asr) == len(lm)
    assert len(lm) == len(uasr)

    ref_lid = [x.strip() for x in open(args.ref_lid, "r").readlines()]
    nbest_lid= [x.strip() for x in open(args.nbest_lid, "r").readlines()]
    ref_asr = [x.strip() for x in open(args.ref_asr, "r").readlines()]
    nbest_asr = [x.strip() for x in open(args.nbest_asr, "r").readlines()]

    assert len(ref_lid) * args.n == len(nbest_lid)
    assert len(ref_asr) * args.n == len(nbest_asr)
    assert len(ref_lid) == len(ref_asr)

    lengths = [len(x) for x in nbest_asr]

    feats = [[s, w, a, l, u, le] for s,w,a,l,u,le in zip(slid, wlid, asr, lm, uasr, lengths)]
 
    weights = []
    for i in range(args.iters):
        s_w = np.random.rand() * args.slid_scale
        w_w = np.random.rand() * args.wlid_scale
        a_w = np.random.rand() * args.asr_scale
        l_w = np.random.rand() * args.lm_scale
        u_w = np.random.rand() * args.uasr_scale
        le_w = (np.random.rand() -0.5) * args.len_scale
        weights.append([s_w, w_w, a_w, l_w, u_w, le_w])

    num_tries = len(weights)
    print("Total number of search points", num_tries)
    threads = args.num_jobs
    pool = Pool(threads)
    compute_fxn = partial(compute, feats=feats, ref_lid=ref_asr, nbest_lid=nbest_lid, ref_asr=ref_asr, nbest_asr=nbest_asr, n=args.n, exclude=args.exclude)
    results = pool.map(compute_fxn, weights)
    pool.close()
    pool.join()

    assert len(results) == len(weights)

    wer_best = 100
    best = ""
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)
    with open(args.dst + "/results.all", "w") as f_out:
        for result in results:
            f_out.write(str(result)+"\n")
            if result["asr_wer"] < wer_best:
                wer_best = result["asr_wer"]
                best = result

    with open(args.dst + "/best_coefficients", "w") as f_out:
        f_out.write(str(best)+"\n")