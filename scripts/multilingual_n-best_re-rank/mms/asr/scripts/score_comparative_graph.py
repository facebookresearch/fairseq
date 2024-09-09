import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy
import re
from bs4 import BeautifulSoup

def get_ethnologue_info(iso):
    fl = f"/checkpoint/vineelkpratap/data/ethnologue/{iso}.txt"
    if not os.path.exists(fl):
        return "NULL",  "NULL",  "NULL",  "NULL",  "NULL",  "NULL",  "NULL"
    with open(f"/checkpoint/vineelkpratap/data/ethnologue/{iso}.txt") as f:
        soup = BeautifulSoup(f.read())
#     print(soup.text)
    name = soup.find("h1",id= "page-title").text
    all_divs = soup.find_all("div")
    alternate_names = "NULL"
    c1 = "NULL"
    c2 = "NULL"
    pop = "NULL"
    status = "NULL"
    writing = "NULL"
    pop2 = "NULL"
    for i, ad in enumerate(all_divs):
        if ad.text.lower() == "alternate names":
            alternate_names = all_divs[i+1].text.strip()
        elif ad.text.lower() == "classification":
            tree = all_divs[i+1].text.split(",")
            if len(tree) > 0:
                c1 = tree[0].strip()
            if len(tree) > 1:
                c2 = tree[1].strip()
        elif ad.text.lower() == "user population":
            pop = all_divs[i+1].text.strip().replace("\n", " ")
            all_pops = re.findall(r'\d+', pop.replace(",", "").replace(".", " "))
            all_pops.append(0)
            final_pop = max([int(ap) for ap in all_pops])
            pop2 = final_pop
        elif ad.text.lower() == "language status":
            status = all_divs[i+1].text.strip().replace("\n", " ").split("We use an asterisk")[0]
        elif ad.text.lower() == "writing":
            writing = all_divs[i+1].text.strip().replace("\n", " ")
    # return iso, name, alternate_names, c1, c2, writing, status, pop, pop2

    if writing != "NULL":
        toks = writing.split(".")
        # print(iso, toks)
        import pdb; pdb.set_trace()
        for t in toks:
            if "primary usage" in t:
                return t.strip()

    return writing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--hyp_ora', type=str)  # asr
    parser.add_argument('--hyp_dep', type=str)  # asr
    parser.add_argument('--ref', type=str)  # asr
    parser.add_argument('--ref_lid', type=str, default=None)  # ref lid
    parser.add_argument('--hyp_lid', type=str, default=None)  # hyp lid
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    parser.add_argument('--script', type=int, default=0)
    parser.add_argument('--lowercase', type=int, default=0)
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

    # tracking delta(ora, dep)
    total_errors = 0
    total_length = 0

    # utt level log, later sorted by delta
    data = []

    for i, (hyp_ora, hyp_dep, ref) in enumerate(zip(hyps_ora, hyps_dep, refs)):
        if args.exclude is not None:
            if langs[i] in args.exclude:
                continue
        
        if args.lowercase != 0:
            hyp_ora = hyp_ora.lower()
            hyp_dep = hyp_dep.lower()
            ref = ref.lower()

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
        exclude_tag = ".exclude_" + "_".join(args.exclude)
    else:
        exclude_tag = ""

    if args.lowercase == 1:
        lc_tag = ".lc"
    else:
        lc_tag = ""

    with open(args.hyp_dep + ".result.comparative.graph" + exclude_tag + lc_tag, "w") as f:
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
                f.write("PAIR\tORACLE\tLID-DEP\tDELTA\tCOUNT\tSAME-SCRIPT\n")
                langs_sorted = sorted([k for k in conf_errors.keys()], key=lambda x: conf_errors[x] / conf_length[x])
                for l in langs_sorted:
                    lang1, lang2 = l.split("-->",1)
                    if args.script == 0:
                        script_change = "n/a"
                    else:
                        script_change = int(get_ethnologue_info(lang1) == get_ethnologue_info(lang2))
                    f.write(l + "\t" + '{:.4g}'.format(conf_errors_ora[l] * 100 / conf_length[l]) + "\t" + \
                                    '{:.4g}'.format(conf_errors_dep[l] * 100 / conf_length[l]) + "\t" + \
                                    '{:.4g}'.format(conf_errors[l] * -100 / conf_length[l]) + "\t" + \
                                    '{:.4g}'.format(conf_count[l]) + "\t" + \
                                    str(script_change) + "\n")