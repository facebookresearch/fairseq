import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--hyp', type=str)
    parser.add_argument('--score', type=str)
    parser.add_argument('--ref_lid', type=str)
    parser.add_argument('--hyp_lid', type=str)
    parser.add_argument('--topk_lid', type=str)
    parser.add_argument('--k', type=str, default=10)
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    args = parser.parse_args()

    hyps = [x for x in open(args.hyp, "r").readlines()]
    langs = [x for x in open(args.ref_lid, "r").readlines()]
    confs = [x for x in open(args.hyp_lid, "r").readlines()]
    topk_lid = [x for x in open(args.topk_lid, "r").readlines()]

    score = [float(x.strip()) for x in open(args.score, "r").readlines()]
    assert len(hyps) == len(score)
    assert len(topk_lid) == len(score)
    assert len(hyps) // args.k == len(langs)
    assert len(hyps) // args.k == len(confs)

    assert len(hyps) % args.k == 0

    ks = [1,2,3,4,5,6,7,8,9,10]
    
    with open(args.score + ".stats", "w") as f2:
        f2.write("K\tTOTAL_ACC\tCORRECT_SUBSET_ACC\tERROR_SUBSET_ACC\n")
        for k in ks:
            res = []
            correct = 0
            err_correct = 0
            corr_correct = 0
            total = 0
            err_total = 0
            corr_total = 0

            for i in range(len(hyps) // args.k):
                idx = i * args.k 
                cands = hyps[idx: idx + k]
                s = score[idx: idx + k]
                cands_lid = topk_lid[idx: idx + k]

                try:
                    sel, max_val = max(enumerate(s), key=lambda x: x[1])
                except:
                    import pdb;pdb.set_trace()

                sel_lid = cands_lid[sel]

                res.append(cands[sel])

                if args.exclude is not None:
                    if langs[i] in args.exclude:
                        continue

                total += 1
                if langs[i] == confs[i]:
                    corr_total += 1
                    if sel_lid == langs[i]:
                        corr_correct += 1
                        correct += 1
                else:
                    err_total += 1
                    if sel_lid == langs[i]:
                        err_correct += 1
                        correct += 1
                    

            assert len(res) == len(hyps) // args.k

            with open(args.hyp + ".sel_score.k" + str(k), "w") as f:
                f.writelines(res)

            f2.write(str(k) + "\t" + '{:.4g}'.format(correct * 100 / total) + "\t" + '{:.4g}'.format(corr_correct * 100 / corr_total) + "\t" + '{:.4g}'.format(err_correct * 100 / err_total) + "\n" )        