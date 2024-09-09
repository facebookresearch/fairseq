import argparse
import json
from collections import defaultdict
import os
import soundfile as sf
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--exp', type=str)
    parser.add_argument('--dump', type=str)
    args = parser.parse_args()

    langs = [d for d in os.listdir(args.dump) if os.path.isdir(os.path.join(args.dump, d))]

    data = {}

    for lang in tqdm(langs):
        print(lang)
        ids = [int(x.strip()) for x in open(args.dump + "/" + lang + "/ids.txt", "r").readlines()]
        # word_hyps = [x.strip() for x in open(args.exp + "/" + lang + "/hypo.word.reord", "r").readlines()]
        # unit_hyps = [x.strip() for x in open(args.exp + "/" + lang + "/hypo.units.reord", "r").readlines()]
        scores = [x.strip() for x in open(args.exp + "/" + lang + "/falign.reord", "r").readlines()]
        # ps = [x for x in open(args.exp + "/" + lang + "/p.reord", "r").readlines()]
        # assert len(ids) == len(word_hyps)
        # assert len(ids) == len(unit_hyps)
        assert len(ids) == len(scores)
        # assert len(ids) == len(ps)
        for id, s in zip(ids, scores):
            if id in data:
                print("Duplicate ID found")
                import pdb;pdb.set_trace()
            data[id] = s

    with open(args.exp + "/falign.score", "w") as f1:
        for i in range(len(data.keys())):
            f1.write(data[i] + "\n")