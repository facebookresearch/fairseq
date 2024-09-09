import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--lids', type=str)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--mapping', type=str, default=None)
    parser.add_argument('--ft', type=str, default='0')  # comes from fasttext, do extra processing
    args = parser.parse_args()

    lids = [x.strip() for x in open(args.lids, "r").readlines()]
    refs = [x.strip() for x in open(args.ref, "r").readlines()]
    if args.mapping is not None:
        # whisper_lid_code:mms_lid_code
        mapping = {x[0]:x[1] for x in [l.strip().split(":", 1) for l in open(args.mapping, "r").readlines()]}
    else:
        mapping = None

    # agg stats
    total = 0
    correct = 0
    topk_correct = 0

    # per lang stats
    lang_total = defaultdict(int)
    lang_correct = defaultdict(int)
    # lang_topk_correct = defaultdict(int)
    confusions = defaultdict(lambda: defaultdict(int))

    # with open(args.expdir + "/lid.txt", "w") as f1, open(args.expdir + "/correct_flag.txt", "w") as f2:
    for i, lid in enumerate(lids):
        ref = refs[i]

        # if pred == "n/a":   # just consider it correct
        #     f1.write(ref + "\n")
        #     f2.write(str(1) + "\n")
        #     continue
        
        # try:
        #     data = json.loads(pred)
        # except:
        #     data = eval(pred)

        if args.ft == '1':
            data = [(x[0].split("_")[-2], x[1]) for x in data]
        
        # if mapping is not None:
        #     topk = [mapping[x[0]] for x in data]
        # else:
        #     topk = [x[0] for x in data]
        top1 = lid
        # topk = set(topk)
        total += 1
        lang_total[ref] += 1
        # f1.write(top1 + "\n")
        # f2.write(str(int(top1 == ref)) + "\n")

        if top1 == ref:
            correct += 1
            lang_correct[ref] += 1
            # topk_correct += 1
            # lang_topk_correct[ref] += 1
        # elif ref in topk:
        #     topk_correct += 1
        #     lang_topk_correct[ref] += 1

        if top1 != ref:
            confusions[ref][top1] += 1

    with open(args.lids + ".result.txt", "w") as f, open(args.lids + ".confusions.txt", "w") as f2:
        f.write("AGGREGATE RESULTS\n")
        f.write("acc: " + '{:.4g}'.format(correct / total) + "\n")
        # f.write("topk_acc: " + '{:.4g}'.format(topk_correct / total) + "\n")

        f.write("\nPER LANG RESULTS\n")
        f.write("lang\tacc\n")
        for k in lang_total.keys():
            f.write(k + "\t" + \
                '{:.4g}'.format(lang_correct[k] / lang_total[k]) + "\t" + "\n")
            
            f2.write(k + "\t" + "\t".join([c + ":" + '{:.4g}'.format(confusions[k][c] / lang_total[k]) for c in confusions[k].keys()]) + "\n")