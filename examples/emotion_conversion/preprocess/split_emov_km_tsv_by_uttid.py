from pathlib import Path
import os
import sys
import argparse
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from build_translation_manifests import get_utt_id


def train_val_test_split(tsv_lines, km_lines, valid_percent, test_percent, seed=42):
    utt_ids = list(sorted(set([get_utt_id(x) for x in tsv_lines])))
    utt_ids, valid_utt_ids, _, _ = train_test_split(utt_ids, utt_ids, test_size=valid_percent, shuffle=True, random_state=seed)
    train_utt_ids, test_utt_ids, _, _ = train_test_split(utt_ids, utt_ids, test_size=test_percent, shuffle=True, random_state=seed)

    train_idx = [i for i, line in enumerate(tsv_lines) if get_utt_id(line) in train_utt_ids]
    valid_idx = [i for i, line in enumerate(tsv_lines) if get_utt_id(line) in valid_utt_ids]
    test_idx = [i for i, line in enumerate(tsv_lines) if get_utt_id(line) in test_utt_ids]

    train_tsv, train_km = [tsv_lines[i] for i in train_idx], [km_lines[i] for i in train_idx]
    valid_tsv, valid_km = [tsv_lines[i] for i in valid_idx], [km_lines[i] for i in valid_idx]
    test_tsv, test_km = [tsv_lines[i] for i in test_idx], [km_lines[i] for i in test_idx]

    print(f"train {len(train_km)}")
    print(f"valid {len(valid_km)}")
    print(f"test {len(test_km)}")

    return train_tsv, train_km, valid_tsv, valid_km, test_tsv, test_km


if __name__ == "__main__":
    """
    this is a standalone script to process a km file
    specifically, to dedup or remove tokens that repeat less
    than k times in a row
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("tsv", type=str, help="path to tsv file")
    parser.add_argument("km", type=str, help="path to km file")
    parser.add_argument("--destdir", required=True, type=str)
    parser.add_argument("--valid-percent", type=float, default=0.05, help="percent to allocate to validation set")
    parser.add_argument("--test-percent", type=float, default=0.05, help="percent to allocate to test set")
    parser.add_argument("--seed", type=int, default=42, help="")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.destdir, exist_ok=True)
    km = open(args.km, "r").readlines()
    tsv = open(args.tsv, "r").readlines()
    root, tsv = tsv[0], tsv[1:]

    assert args.tsv.endswith(".tsv") and args.km.endswith(".km")
    assert len(tsv) == len(km)

    train_tsv, train_km, valid_tsv, valid_km, test_tsv, test_km = train_val_test_split(tsv, km, args.valid_percent, args.test_percent, args.seed)

    assert len(train_tsv) + len(valid_tsv) + len(test_tsv) == len(tsv)
    assert len(train_tsv) == len(train_km) and len(valid_tsv) == len(valid_km) and len(test_tsv) == len(test_km)

    dir = Path(args.destdir)
    open(dir / f"train.tsv", "w").writelines([root] + train_tsv)
    open(dir / f"valid.tsv", "w").writelines([root] + valid_tsv)
    open(dir / f"test.tsv", "w").writelines([root] + test_tsv)
    open(dir / f"train.km", "w").writelines(train_km)
    open(dir / f"valid.km", "w").writelines(valid_km)
    open(dir / f"test.km", "w").writelines(test_km)
    print("done")
