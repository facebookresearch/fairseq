import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
import subprocess
import re
import pandas as pd

# https://huggingface.co/datasets/espnet/long-yodas-segmented

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()

    # load parquets, write audio to dir/processed/lang/audio/*.wavs and text to dir/processed/lang/text
    # text = {uttid, videoid, lang, score, wavpath, text}

    parquets = [args.dir + "/data/" + d for d in os.listdir(args.dir + "/data/")]
    text_data = defaultdict(list)

    for p in tqdm(parquets):
        df = pd.read_parquet(p)
        for index, row in df.iterrows():
            # save text, write later
            text = {"utt_id":row['utt_id'], "video_id":row["video_id"], "language":row["language"], "score":row["score"], "text":row["text"]}
            

            # write audio
            if not os.path.exists(args.dir + "/processed/" + text["language"] + "/audio/"):
                os.makedirs(args.dir + "/processed/" + text["language"] + "/audio/")

            wav_path = args.dir + "/processed/" + text["language"] + "/audio/" + text["utt_id"] + ".wav"
            # with open(wav_path, "bx") as f:
            #     f.write(row["audio"]["bytes"])

            text["wav_path"] = wav_path
            text_data[text["language"]].append(text)
        
    for lang in text_data.keys():
        with open(args.dir + "/processed/" + lang + "/text", "w") as f:
            f.writelines([str(x) + "\n" for x in text_data[lang]])