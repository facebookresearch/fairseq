import argparse
import json
from collections import defaultdict
import os
import soundfile as sf
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--wavs', type=str)
    parser.add_argument('--lids', type=str)
    parser.add_argument('--dst', type=str, default=None)
    args = parser.parse_args()

    # split wavs into dst/lang/wav.txt and dst/lang/ids.txt

    wavs = [x.strip() for x in open(args.wavs, "r").readlines()]
    lids = [x.strip() for x in open(args.lids, "r").readlines()]

    assert len(wavs) == len(lids)

    lang_split = defaultdict(list)
    for id, (wav,lid) in enumerate(zip(wavs, lids)):
        lang_split[lid].append((id, wav))

    for lang in tqdm(lang_split.keys()):
        if not os.path.exists(args.dst + "/" + lang):
            os.makedirs(args.dst + "/" + lang)

        with open(args.dst + "/" + lang + "/test.tsv", "w") as f1, \
            open(args.dst + "/" + lang + "/ids.txt", "w") as f2:
            f1.write("/\n")
            f1.writelines([x[1] + "\t" + str(sf.SoundFile(x[1]).frames) + "\n" for x in lang_split[lang]])
            f2.writelines([str(x[0]) + "\n" for x in lang_split[lang]])

        with open(args.dst + "/" + lang + "/test.ltr", "w") as fw:
            fw.write("d u m m y | d u m m y |\n"*len(lang_split[lang]))
        with open(args.dst + "/" + lang + "/test.wrd", "w") as fw:
            fw.write("dummy dummy\n"*len(lang_split[lang]))