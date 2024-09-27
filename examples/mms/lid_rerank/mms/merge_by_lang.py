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

    for lang in langs:
        ids = [int(x.strip()) for x in open(args.dump + "/" + lang + "/ids.txt", "r").readlines()]
        word_hyps = [x.strip() for x in open(args.exp + "/" + lang + "/hypo.word.reord", "r").readlines()]
        scores = [x.strip() for x in open(args.exp + "/" + lang + "/asr_score.reord", "r").readlines()]
        assert len(ids) == len(word_hyps)
        assert len(ids) == len(scores)
        for id, word_hyp, s in zip(ids, word_hyps, scores):
            if id in data:
                print("Duplicate ID found")
                import pdb;pdb.set_trace()
            data[id] = (word_hyp, s)

    with open(args.exp + "/nbest_asr_hyp", "w") as f1, open(args.exp + "/asr_score", "w") as f2:
        for i in range(len(data.keys())):
            f1.write(data[i][0] + "\n")
            f2.write(data[i][1] + "\n")