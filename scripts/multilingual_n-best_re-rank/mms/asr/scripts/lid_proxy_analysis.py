# Can we use LID accuracy as a proxy for ASR accuracy (in case LID would be correct)? Or is LID accuracy a good way to identify examples that are also hard for ASR?

import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy
import statistics

def bucket(scores):
    # [0-20, 20-40, 40-60, 60-80, 80-100, 100+]
    b = [0, 0, 0, 0, 0, 0]
    for s in scores:
        if s < .2:
            b[0] += 1
        elif s < .4:
            b[1] += 1
        elif s < .6:
            b[2] += 1
        elif s < .8:
            b[3] += 1
        elif s < 1:
            b[4] += 1
        else:
            b[5] += 1
    return [x / len(scores) for x in b]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--hyp', type=str)  # asr
    parser.add_argument('--ref', type=str)  # asr
    parser.add_argument('--ref_lid', type=str)  # ref lid
    parser.add_argument('--hyp_lid', type=str)  # hyp lid
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    parser.add_argument('--lc', type=int, default=0)  # lowercase
    parser.add_argument('--rm', type=int, default=0)  # remove punc
    parser.add_argument('--wer', type=int, default=0)  # WER
    args = parser.parse_args()

    hyps = [x.strip() for x in open(args.hyp, "r").readlines()]
    refs = [x.strip() for x in open(args.ref, "r").readlines()]
    assert len(hyps) == len(refs)

    langs = [x.strip() for x in open(args.ref_lid, "r").readlines()]
    assert len(langs) == len(hyps)

    confusions = [x.strip() for x in open(args.hyp_lid, "r").readlines()]
    assert len(confusions) == len(hyps)

    if args.wer != 0:
        cer_langs = [x.strip() for x in open("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/cer_langs.txt", "r").readlines()]

    
    data = []   # num_utts * (is_lid_correct, wer)

    for i, (hyp, ref) in tqdm(enumerate(zip(hyps, refs))):
        if args.exclude is not None:
            if langs[i] in args.exclude:
                continue

        if args.lc != 0:
            hyp = hyp.lower()
            ref = ref.lower()
        
        if args.rm != 0:
            hyp = hyp.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
            ref = ref.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")

        if args.wer != 0 and langs[i] in cer_langs:
            hyp = " ".join(hyp)
            ref = " ".join(ref)

        hyp_words = hyp.split()
        tgt_words = ref.split()
        
        # skip empty refs
        if ref == "":
            continue
        if ref.strip() == "":
            continue

        errs = editdistance.eval(hyp_words, tgt_words)
        # import pdb;pdb.set_trace()

        if langs[i] == confusions[i]:
            is_lid_correct = 1
        else:
            is_lid_correct = 0

        data.append([is_lid_correct, errs/len(tgt_words)])
    
    if args.exclude is not None:
        if len(args.exclude) > 50:
            exclude_tag = ".exclude_" + str(len(args.exclude))
        else:
            exclude_tag = ".exclude_" + "_".join(args.exclude)
    else:
        exclude_tag = ""

    if args.lc == 1:
        lc_tag = ".lc"
    else:
        lc_tag = ""
    
    if args.rm == 1:
        rm_tag = ".rm"
    else:
        rm_tag = ""

    with open(args.hyp + ".proxy_analysis" + exclude_tag + lc_tag + rm_tag, "w") as f:
        f.writelines([str(x[0]) + "\t" + str(x[1]) + "\n" for x in data])

    correct = [x[1] for x in data if x[0] == 1]
    incorrect = [x[1] for x in data if x[0] == 0]

    print("Correct")
    print("mean:", str(statistics.mean(correct)))
    print("stdev:", str(statistics.stdev(correct)))
    print("buckets")
    for x in bucket(correct):
        print(str(x))

    print("\nInorrect")
    print("mean:", str(statistics.mean(incorrect)))
    print("stdev:", str(statistics.stdev(incorrect)))
    print("buckets")
    for x in bucket(incorrect):
        print(str(x))

