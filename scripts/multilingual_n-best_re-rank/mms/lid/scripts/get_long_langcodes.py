import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--langs', type=str)    # data/*.html
    args = parser.parse_args()

    mapping = {"cmn":"cmn-script_simplified", "srp":"srp-script_latin", "urd":"urd-script_arabic", "uzb":"uzb-script_latin", "yue":"yue-script_traditional", "aze":"azj-script_latin", "kmr":"kmr-script_latin"}
    mapping = {mapping[k]:k for k in mapping.keys()}

    html_lines = open(args.langs, "r").readlines()
    langs = []
    for l in html_lines:
        toks = l.split()
        if toks[0] == "<p>" and toks[1] != "Iso":
            if toks[1] in mapping:
                lang = mapping[toks[1]]
            else:
                lang = toks[1]
            langs.append(lang)
    langs = set(langs)
    long_langcodes = [x for x in langs if "-" in x]
    import pdb;pdb.set_trace()