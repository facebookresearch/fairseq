import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--ref_lid', type=str)  # ref lid
    parser.add_argument('--hyp_lid', type=str)  # hyp lid
    parser.add_argument('--topk_lid', type=str)  # topk lid
    parser.add_argument('--preds', type=str)  # written lid preds
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    args = parser.parse_args()

    langs = [x.strip() for x in open(args.ref_lid, "r").readlines()]

    confusions = [x.strip() for x in open(args.hyp_lid, "r").readlines()]
    assert len(confusions) == len(langs)

    topk_langs = [x.strip() for x in open(args.topk_lid, "r").readlines()]
    assert len(topk_langs) == len(langs) * args.k

    preds = [x.strip() for x in open(args.preds, "r").readlines()]
    assert len(preds) == len(topk_langs)

    # gt_correct = 0  # equals lang
    # conf_correct = 0    # equals confusions
    # total = 0

    k_gt_correct = defaultdict(int)
    k_conf_correct = defaultdict(int)
    k_total = defaultdict(int)

    subset_k_gt_correct = defaultdict(int)
    subset_k_conf_correct = defaultdict(int)
    subset_k_total = defaultdict(int)

    num_utts = 0
    agreement = []
    top1 = []
    scores = []
    for i, pred in tqdm(enumerate(preds)):
        if args.exclude is not None:
            if langs[i // args.k] in args.exclude:
                continue

        which_k = i % args.k
        pred = eval(pred)
        # pred_1 = pred[0][0].split("_")[-2]
        pred_1 = pred[0][0]
        top1.append(pred_1)

        try:
            pred_all = [x[0] for x in pred]
            idx = pred_all.index(topk_langs[i])
            score = pred[idx][-1]
        except:
            score = 0
        scores.append(score)
        # import pdb;pdb.set_trace()


        if pred_1 == langs[i // args.k]:
            # gt_correct += 1
            k_gt_correct[which_k] += 1
        if pred_1 == topk_langs[i]:
            # conf_correct += 1
            k_conf_correct[which_k] += 1
            agreement.append(1)
        else:
            agreement.append(0)
        # total += 1
        k_total[which_k] += 1

        if langs[i // args.k] != confusions[i // args.k]:
            num_utts += 1
            if pred_1 == langs[i // args.k]:
                subset_k_gt_correct[which_k] += 1
            if pred_1 == topk_langs[i]:
                subset_k_conf_correct[which_k] += 1
            subset_k_total[which_k] += 1
    
    print(num_utts)
    
    # import pdb;pdb.set_trace()
    with open(args.preds + ".result", "w") as f:
        # f.write("GT ACC\n" + '{:.4g}'.format(gt_correct / total) + "\n")
        # f.write("\nCONF ACC\n" + '{:.4g}'.format(conf_correct / total) + "\n")

        f.write("GT ACC BY K\n")
        for k in k_total:
            f.write(str(k) + "\t" + '{:.4g}'.format(k_gt_correct[k] * 100 / k_total[k]) + "\n")

        f.write("\nSUBSET GT ACC BY K\n")
        for k in subset_k_total:
            f.write(str(k) + "\t" + '{:.4g}'.format(subset_k_gt_correct[k] * 100 / subset_k_total[k]) + "\n")

        f.write("\nCONF ACC BY K\n")
        for k in k_total:
            f.write(str(k) + "\t" + '{:.4g}'.format(k_conf_correct[k] * 100 / k_total[k]) + "\n")

        f.write("\nSUBSET CONF ACC BY K\n")
        for k in subset_k_total:
            f.write(str(k) + "\t" + '{:.4g}'.format(subset_k_conf_correct[k] * 100 / subset_k_total[k]) + "\n")

    with open(args.preds + ".agreement", "w") as f:
        f.writelines([str(x) + "\n" for x in agreement])

    with open(args.preds + ".top1", "w") as f:
        f.writelines([str(x) + "\n" for x in top1])

    with open(args.preds + ".scores", "w") as f:
        f.writelines([str(x) + "\n" for x in scores])