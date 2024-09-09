import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--langs', type=str)
    parser.add_argument('--mapping', type=str)
    parser.add_argument('--set', type=str)
    args = parser.parse_args()

    langs = [x.strip() for x in open(args.langs, "r").readlines()]
    with open(args.mapping, 'r') as f:
        mapping = json.load(f)['lid']['fl']
    all_tsv = ["/"]
    all_lang = []
    all_ltr = []
    all_wrd = []
    
    for l in langs:
        # lang
        lang_code = mapping[l][0]

        # tsv
        src_dir = args.src + "/" + l + "/audio/" + args.set + "/"
        lang_tsv = [x.strip() for x in open(args.src + "/manifests/" + l + "/fseq_pristine/" + args.set + ".tsv", "r").readlines()]
        for t in lang_tsv[1:]:
            all_tsv.append(src_dir + t)
            all_lang.append(lang_code)

        # ltr
        lang_ltr = [x.strip() for x in open(args.src + "/manifests/" + l + "/fseq_pristine/" + args.set + ".ltr", "r").readlines()]
        for t in lang_ltr:
            all_ltr.append(t)
        # wrd
        lang_wrd = [x.strip() for x in open(args.src + "/manifests/" + l + "/fseq_pristine/" + args.set + ".wrd", "r").readlines()]
        for t in lang_wrd:
            all_wrd.append(t)
        
    with open(args.dst + "/" + args.set + ".tsv", "w") as f:
        f.writelines([x + "\n" for x in all_tsv])
    
    with open(args.dst + "/" + args.set + ".lang", "w") as f:
        f.writelines([x + "\n" for x in all_lang])

    with open(args.dst + "/" + args.set + ".ltr", "w") as f:
        f.writelines([x + "\n" for x in all_ltr])

    with open(args.dst + "/" + args.set + ".wrd", "w") as f:
        f.writelines([x + "\n" for x in all_wrd])