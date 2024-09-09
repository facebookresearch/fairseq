import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy
import re

def get_dict(dict_txt):
    rv = [x.strip().split() for x in open(dict_txt, "r").readlines()]
    return {x[0]:int(x[1]) for x in rv}

def analyze_overlap(lang1, lang2, exclude_space=0):
    try:
        dict1 = get_dict("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/dict/" + lang1 + ".txt")
        dict2 = get_dict("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/dict/" + lang2 + ".txt")
    except:
        return "n/a", "n/a"
    # import pdb;pdb.set_trace()
    # raw = len(set(dict1.keys()).intersection(set(dict2.keys()))) / len(set(dict1.keys()).union(set(dict2.keys())))
    overlap = set(dict1.keys()).intersection(set(dict2.keys()))
    raw = len(overlap) / len(set(dict1.keys()))

    if exclude_space != 0:
        weighted = sum([dict1[x] for x in overlap if x != "|"]) / sum([dict1[x] for x in dict1.keys() if x != "|"])
    else:
        weighted = sum([dict1[x] for x in overlap]) / sum([dict1[x] for x in dict1.keys()])
    return raw, weighted
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--hyp_ora', type=str)  # asr
    parser.add_argument('--hyp_dep', type=str)  # asr
    parser.add_argument('--ref', type=str)  # asr
    parser.add_argument('--ref_lid', type=str, default=None)  # ref lid
    parser.add_argument('--hyp_lid', type=str, default=None)  # hyp lid
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    parser.add_argument('--script', type=int, default=0)
    parser.add_argument('--lc', type=int, default=0)
    parser.add_argument('--rm', type=int, default=0)  # remove punc
    parser.add_argument('--wer', type=int, default=0)  # WER
    parser.add_argument('--exclude_space', type=int, default=0)
    args = parser.parse_args()

    hyps_ora = [x.strip() for x in open(args.hyp_ora, "r").readlines()]
    hyps_dep = [x.strip() for x in open(args.hyp_dep, "r").readlines()]
    refs = [x.strip() for x in open(args.ref, "r").readlines()]
    assert len(hyps_ora) == len(refs)
    assert len(hyps_dep) == len(refs)

    if args.ref_lid is not None:
        langs = [x.strip() for x in open(args.ref_lid, "r").readlines()]
        assert len(langs) == len(hyps_ora)
        lang_errors = defaultdict(int)
        lang_errors_ora = defaultdict(int)
        lang_errors_dep = defaultdict(int)
        lang_count = defaultdict(int)
        lang_length = defaultdict(int)

        if args.hyp_lid is not None:
            confusions = [x.strip() for x in open(args.hyp_lid, "r").readlines()]
            assert len(confusions) == len(hyps_ora)
            conf_errors = defaultdict(int)
            conf_errors_ora = defaultdict(int)
            conf_errors_dep = defaultdict(int)
            conf_count = defaultdict(int)
            conf_length = defaultdict(int)
    else:
        langs = None
        confusions = None

    if args.wer != 0:
        cer_langs = [x.strip() for x in open("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/cer_langs.txt", "r").readlines()]

    # tracking delta(ora, dep)
    total_errors = 0
    total_length = 0

    # utt level log, later sorted by delta
    data = []

    for i, (hyp_ora, hyp_dep, ref) in tqdm(enumerate(zip(hyps_ora, hyps_dep, refs))):
        if args.exclude is not None:
            if langs[i] in args.exclude:
                continue
        
        if args.lc != 0:
            hyp_ora = hyp_ora.lower()
            hyp_dep = hyp_dep.lower()
            ref = ref.lower()

        if args.rm != 0:
            hyp_ora = hyp_ora.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
            hyp_dep = hyp_dep.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
            ref = ref.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")

        if args.wer != 0 and langs[i] in cer_langs:
            hyp_ora = " ".join(hyp_ora)
            hyp_dep = " ".join(hyp_dep)
            ref = " ".join(ref)

        if args.exclude_space != 0:
            hyp_ora = hyp_ora.replace("|","")
            hyp_dep = hyp_dep.replace("|","")
            ref = ref.replace("|","")

        tgt_words = ref.split()

        # skip empty refs
        if ref == "":
            continue

        # check sub/ins/del with another pkg
        try:
            summary_ora = werpy.summary(ref, hyp_ora)
            summary_dep = werpy.summary(ref, hyp_dep)
        except:
            import pdb;pdb.set_trace()

        total_errors += summary_ora.ld[0] - summary_dep.ld[0]
        total_length += len(tgt_words)
        
        if langs is not None:
            lang_errors[langs[i]] += summary_ora.ld[0] - summary_dep.ld[0]
            lang_errors_ora[langs[i]] += summary_ora.ld[0]
            lang_errors_dep[langs[i]] += summary_dep.ld[0]
            lang_count[langs[i]] += 1
            lang_length[langs[i]] += len(tgt_words)

            if confusions is not None:
                pair = langs[i] + "-->" + confusions[i]
                conf_errors[pair] += summary_ora.ld[0] - summary_dep.ld[0]
                conf_errors_ora[pair] += summary_ora.ld[0]
                conf_errors_dep[pair] += summary_dep.ld[0]
                conf_count[pair] += 1
                conf_length[pair] += len(tgt_words)
    
    # import pdb;pdb.set_trace()
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

    with open(args.hyp_dep + ".result.comparative.graph" + exclude_tag + lc_tag + rm_tag, "w") as f:
        if langs is not None:
            f.write("\nPER LANG ERROR RATES\n")
            f.write("LANG\tORACLE\tLID-DEP\tDELTA\tCOUNT\n")
            langs_sorted = sorted([k for k in lang_errors.keys()], key=lambda x: lang_errors[x] / lang_length[x])
            for l in langs_sorted:
                f.write(l + "\t" + '{:.4g}'.format(lang_errors_ora[l] * 100 / lang_length[l]) + "\t" + \
                                    '{:.4g}'.format(lang_errors_dep[l] * 100 / lang_length[l]) + "\t" + \
                                    '{:.4g}'.format(lang_errors[l] * -100 / lang_length[l]) + "\t" + \
                                    '{:.4g}'.format(lang_count[l]) + "\n")

            if confusions is not None:
                f.write("\nPER CONFUSION PAIR ERROR RATES\n")
                f.write("PAIR\tORACLE\tLID-DEP\tDELTA\tCOUNT\tSCRIPT-OVERLAP\tSCRIPT-OVERLAP-WEIGHTED\n")
                langs_sorted = sorted([k for k in conf_errors.keys()], key=lambda x: conf_errors[x] / conf_length[x])
                for l in langs_sorted:
                    lang1, lang2 = l.split("-->",1)
                    if args.script == 0:
                        script_overlap = "n/a"
                    else:
                        script_overlap, script_overlap_weighted = analyze_overlap(lang1, lang2, args.exclude_space)

                    f.write(l + "\t" + '{:.4g}'.format(conf_errors_ora[l] * 100 / conf_length[l]) + "\t" + \
                                    '{:.4g}'.format(conf_errors_dep[l] * 100 / conf_length[l]) + "\t" + \
                                    '{:.4g}'.format(conf_errors[l] * -100 / conf_length[l]) + "\t" + \
                                    '{:.4g}'.format(conf_count[l]) + "\t" + \
                                    str(script_overlap) + "\t" + str(script_overlap_weighted) + "\n")