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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    dict_files = [f for f in os.listdir(args.src)]

    dicts = {}
    all_chars = set()
    for f in dict_files:
        tmp = get_dict(args.src + "/" + f)
        # normalize
        denom = sum([tmp[x] for x in tmp])
        tmp = {x:tmp[x]/denom for x in tmp}

        all_chars = all_chars.union(set(tmp.keys()))
        lang = f.split(".")[0]
        dicts[lang] = tmp
    
    vectors_bin = {}
    vectors_freq = {}
    for l in dicts.keys():
        v_bin = []    # 1 or 0
        v_freq = []   # based on freq
        for c in all_chars:
            if c in dicts[l]:
                v_bin.append(1)
                v_freq.append(dicts[l][c])
            else:
                v_bin.append(0)
                v_freq.append(0)
        assert len(v_bin) == len(all_chars)
        assert len(v_freq) == len(all_chars)
        vectors_bin[l] = v_bin
        vectors_freq[l] = v_freq

    # import pdb;pdb.set_trace()
    with open(args.dst + "/bin.txt", "w") as f1, \
        open(args.dst + "/freq.txt", "w") as f2, \
        open(args.dst + "/langs.txt", "w") as f3:
        for l in vectors_bin.keys():
            f1.write("\t".join([str(x) for x in vectors_bin[l]]) + "\n")
            f2.write("\t".join([str(x) for x in vectors_freq[l]]) + "\n")
            f3.write(l+"\n")
            