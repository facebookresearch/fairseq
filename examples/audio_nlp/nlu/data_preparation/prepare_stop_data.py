import pandas as pd
from typing import List, Dict
import os

"""
Prepare the STOP dataset in fairseq training format for seq2seq models
    train.utterance, train.seqlogical
    {split}.utterance, {split}.seqlogical
"""

STOP_MANIFEST_DIR: str = "/fsx/akshats/data/stop_updated/"
DOMAINS: List[str] = ["alarm", "event", "messaging", "music", "navigation", "reminder", "timer", "weather"]
MANIFEST_FILE: str = "manifest.tsv"
COLUMNS: List[str] = ["audio_file", "gender", "native", "domain", "utterance", "seqlogical"]

OUTPUT_DIR: str =  "/fsx/akshats/data/stop_fairseq_seq2seq/full_dataset"
OUTPUT_FORMAT: str = "{split}.{lang}"

def save_fairseq_split(stop_df: pd.DataFrame, split: str) -> None:
    subset_df = stop_df[stop_df["split"] == split]
    utterances = subset_df["utterance"]
    seqlogicals = subset_df["seqlogical"]
    output_utterance_file: str = os.path.join(OUTPUT_DIR, OUTPUT_FORMAT.format(split=split, lang="utterance"))
    output_seqlogical_file: str = os.path.join(OUTPUT_DIR, OUTPUT_FORMAT.format(split=split, lang="seqlogical"))

    # write utterances
    with open(output_utterance_file, "w+") as fp:
        print(utterances.head(2))
        utterances.to_csv(fp, index=None, header=None)
        print(f"Wrote to {output_utterance_file}")
    
    # write seqlogicals
    with open(output_seqlogical_file, "w+") as fp:
        print(seqlogicals.head(2))
        seqlogicals.to_csv(fp, index=None, header=None)
        print(f"Wrote to {output_seqlogical_file}")


def get_full_stop_df(debug: bool) -> pd.DataFrame:
    # read all stop files
    all_stop_dfs: List[pd.DataFrame] = []
    for split_category in SPLITS:
        for split_original in SPLITS[split_category]:
            for domain in DOMAINS:
                stop_file = os.path.join(STOP_MANIFEST_DIR, split_original, f'{domain}_{split_category}',MANIFEST_FILE)
                stop_df = pd.read_csv(stop_file, names=COLUMNS, sep="\t")
                stop_df["split"] = split_category
                stop_df["split_original"] = split_original
                if debug:
                    print(f"STOP_FILE: {stop_file}, {os.path.exists(stop_file)}")
                    print(stop_df.head(2))
                all_stop_dfs.append(stop_df)
    full_stop_df = pd.concat(all_stop_dfs)
    return full_stop_df

def main() -> None:
    # read all stop files
    full_stop_df = get_full_stop_df(debug=False)
    print("Full STOP")
    print(full_stop_df.head(2))

    print("STARTING FILE SAVING")
    # save all the splits
    save_fairseq_split(full_stop_df, split="train")
    save_fairseq_split(full_stop_df, split="eval")
    save_fairseq_split(full_stop_df, split="test")


if __name__ == "__main__":
    main()