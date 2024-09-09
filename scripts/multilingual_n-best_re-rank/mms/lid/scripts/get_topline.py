import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--langs', type=str)    # data/*.html
    parser.add_argument('--refs', nargs="*")
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

    for ref in args.refs:
        ref_lines = [x.strip() for x in open(ref, "r").readlines()]
        count = 0
        not_covered = set()
        for r in ref_lines:
            if r in langs:
                count += 1
            else:
                not_covered.add(r)
        print(ref)
        print('{:.4g}'.format(count / len(ref_lines)))
        print(not_covered)