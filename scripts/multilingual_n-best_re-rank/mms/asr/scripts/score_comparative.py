import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--hyp_ora', type=str)  # asr
    parser.add_argument('--hyp_dep', type=str)  # asr
    parser.add_argument('--ref', type=str)  # asr
    parser.add_argument('--ref_lid', type=str, default=None)  # ref lid
    parser.add_argument('--hyp_lid', type=str, default=None)  # hyp lid
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
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
        lang_sub = defaultdict(int)
        lang_ins = defaultdict(int)
        lang_del = defaultdict(int)
        lang_length = defaultdict(int)

        if args.hyp_lid is not None:
            confusions = [x.strip() for x in open(args.hyp_lid, "r").readlines()]
            assert len(confusions) == len(hyps_ora)
            conf_errors = defaultdict(int)
            conf_sub = defaultdict(int)
            conf_ins = defaultdict(int)
            conf_del = defaultdict(int)
            conf_length = defaultdict(int)
    else:
        langs = None
        confusions = None

    # tracking delta(ora, dep)
    total_errors = 0
    total_sub = 0
    total_ins = 0
    total_del = 0
    total_length = 0

    # utt level log, later sorted by delta
    data = []

    for i, (hyp_ora, hyp_dep, ref) in enumerate(zip(hyps_ora, hyps_dep, refs)):
        if args.exclude is not None:
            if langs[i] in args.exclude:
                continue
        
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
        total_sub += summary_ora.substitutions[0] - summary_dep.substitutions[0]
        total_ins += summary_ora.insertions[0] - summary_dep.insertions[0]
        total_del += summary_ora.deletions[0] - summary_dep.deletions[0]
        total_length += len(tgt_words)
        
        if langs is not None:
            lang_errors[langs[i]] += summary_ora.ld[0] - summary_dep.ld[0]
            lang_sub[langs[i]] += summary_ora.substitutions[0] - summary_dep.substitutions[0]
            lang_ins[langs[i]] += summary_ora.insertions[0] - summary_dep.insertions[0]
            lang_del[langs[i]] += summary_ora.deletions[0] - summary_dep.deletions[0]
            lang_length[langs[i]] += len(tgt_words)

            if confusions is not None:
                pair = langs[i] + "-->" + confusions[i]
                conf_errors[pair] += summary_ora.ld[0] - summary_dep.ld[0]
                conf_sub[pair] += summary_ora.substitutions[0] - summary_dep.substitutions[0]
                conf_ins[pair] += summary_ora.insertions[0] - summary_dep.insertions[0]
                conf_del[pair] += summary_ora.deletions[0] - summary_dep.deletions[0]
                conf_length[pair] += len(tgt_words)

        data.append({"confusion": langs[i] + "-->" + confusions[i], "ora_score": summary_ora.ld[0] * 100 / len(tgt_words), "dep_score": summary_dep.ld[0] * 100 / len(tgt_words), "score_delta": (summary_ora.ld[0] - summary_dep.ld[0]) * 100 / len(tgt_words), "ref": ref, "hyp_ora": hyp_ora, "hyp_dep": hyp_dep, "lang": langs[i]})
        
    with open(args.hyp_dep + ".result.comparative", "w") as f:
        f.write("AGGREGATE ERROR RATE\n")
        f.write("TOT SUB INS DEL\n")
        f.write('{:.4g}'.format(total_errors * 100 / total_length) + " " + \
                '{:.4g}'.format(total_sub * 100 / total_length) + " " + \
                '{:.4g}'.format(total_ins * 100 / total_length) + " " + \
                '{:.4g}'.format(total_del * 100 / total_length) + "\n")

        if langs is not None:
            f.write("\nPER LANG ERROR RATES\n")
            f.write("LANG TOT SUB INS DEL\n")
            for l in lang_errors.keys():
                f.write(l + " " + '{:.4g}'.format(lang_errors[l] * 100 / lang_length[l]) + " " + \
                                    '{:.4g}'.format(lang_sub[l] * 100 / lang_length[l]) + " " + \
                                    '{:.4g}'.format(lang_ins[l] * 100 / lang_length[l]) + " " + \
                                    '{:.4g}'.format(lang_del[l] * 100 / lang_length[l]) + "\n")

            if confusions is not None:
                f.write("\nPER CONFUSION PAIR ERROR RATES\n")
                f.write("PAIR TOT SUB INS DEL\n")
                for l in conf_errors.keys():
                    f.write(l + " " + '{:.4g}'.format(conf_errors[l] * 100 / conf_length[l]) + " " + \
                                        '{:.4g}'.format(conf_sub[l] * 100 / conf_length[l]) + " " + \
                                        '{:.4g}'.format(conf_ins[l] * 100 / conf_length[l]) + " " + \
                                        '{:.4g}'.format(conf_del[l] * 100 / conf_length[l]) + "\n")

    data = sorted(data, key=lambda x: x["score_delta"])
    with open(args.hyp_dep + ".result.comparative.data", "w") as f:
        f.writelines([str(x) + "\n" for x in data])