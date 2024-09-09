import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
import subprocess
import re
import pandas as pd
import random


# https://huggingface.co/datasets/espnet/long-yodas-segmented

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()

    langs = [args.dir + d for d in os.listdir(args.dir)]
    
    for l in langs:
        videos = defaultdict(list)
        text = open(l + "/text", "r").readlines()
        for t in text:
            t_data = eval(t)
            videos[t_data["video_id"]].append(t)
        print(l)
        print(len(videos))
        if len(videos) <= 12:    # ~5 min per video
            # make it all test
            test = sorted(videos)
        else:
            # choose a random 12
            test = random.sample(sorted(videos), 12)
        with open(l + "/text.test", "w", encoding="utf-8") as f:
            for v in test:
                f.writelines(videos[v])