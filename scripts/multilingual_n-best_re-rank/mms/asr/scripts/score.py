import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--hyp', type=str)  # asr
    parser.add_argument('--ref', type=str)  # asr
    parser.add_argument('--ref_lid', type=str, default=None)  # ref lid
    parser.add_argument('--hyp_lid', type=str, default=None)  # hyp lid
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    parser.add_argument('--lc', type=int, default=0)  # lowercase
    parser.add_argument('--rm', type=int, default=0)  # remove punc
    parser.add_argument('--wer', type=int, default=0)  # WER
    args = parser.parse_args()

    hyps = [x.strip() for x in open(args.hyp, "r").readlines()]
    refs = [x.strip() for x in open(args.ref, "r").readlines()]
    assert len(hyps) == len(refs)

    if args.ref_lid is not None:
        langs = [x.strip() for x in open(args.ref_lid, "r").readlines()]
        assert len(langs) == len(hyps)
        lang_errors = defaultdict(int)
        lang_sub = defaultdict(int)
        lang_ins = defaultdict(int)
        lang_del = defaultdict(int)
        lang_length = defaultdict(int)

        if args.hyp_lid is not None:
            confusions = [x.strip() for x in open(args.hyp_lid, "r").readlines()]
            assert len(confusions) == len(hyps)
            conf_errors = defaultdict(int)
            conf_sub = defaultdict(int)
            conf_ins = defaultdict(int)
            conf_del = defaultdict(int)
            conf_length = defaultdict(int)
    else:
        langs = None
        confusions = None

    if args.wer != 0:
        cer_langs = [x.strip() for x in open("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/cer_langs.txt", "r").readlines()]

    total_errors = 0
    total_sub = 0
    total_ins = 0
    total_del = 0
    total_length = 0
    num_utts = 0
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

        # check sub/ins/del with another pkg
        # try:
        #     summary = werpy.summary(ref, hyp)
        # except:
        #     import pdb;pdb.set_trace()
        # if summary.ld[0] != errs:
        #     import pdb;pdb.set_trace()
        # errs = summary.ld[0]

        total_errors += errs
        # total_sub += summary.substitutions[0]
        # total_ins += summary.insertions[0]
        # total_del += summary.deletions[0]
        total_length += len(tgt_words)
        
        if langs is not None:
            lang_errors[langs[i]] += errs
            # lang_sub[langs[i]] += summary.substitutions[0]
            # lang_ins[langs[i]] += summary.insertions[0]
            # lang_del[langs[i]] += summary.deletions[0]
            lang_length[langs[i]] += len(tgt_words)

            if confusions is not None:
                pair = langs[i] + "-->" + confusions[i]
                conf_errors[pair] += errs
                # conf_sub[pair] += summary.substitutions[0]
                # conf_ins[pair] += summary.insertions[0]
                # conf_del[pair] += summary.deletions[0]
                conf_length[pair] += len(tgt_words)

        num_utts += 1
    
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

    with open(args.hyp + ".result" + exclude_tag + lc_tag + rm_tag, "w") as f:
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

    print(num_utts)