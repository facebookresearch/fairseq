from pathlib import Path
import os
import argparse
import random
import numpy as np
from sklearn.utils import shuffle


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
    parser.add_argument("-sh", "--shuffle", action="store_true", help="path to km file")
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

    if args.shuffle:
        tsv, km = shuffle(tsv, km)
        print(f"shuffled")

    N = len(tsv)
    N_tt = int(N * args.test_percent)
    N_cv = int(N * args.valid_percent)
    N_tr = N - N_tt - N_cv

    train_tsv = tsv[:N_tr]
    valid_tsv = tsv[N_tr:N_tr + N_cv]
    test_tsv = tsv[N_tr + N_cv:]
    train_km = km[:N_tr]
    valid_km = km[N_tr:N_tr + N_cv]
    test_km = km[N_tr + N_cv:]

    assert len(train_tsv) + len(valid_tsv) + len(test_tsv) == len(tsv)
    assert len(train_tsv) == len(train_km) and len(valid_tsv) == len(valid_km) and len(test_tsv) == len(test_km)

    dir = Path(args.destdir)
    open(dir / f"train.tsv", "w").writelines([root] + train_tsv)
    open(dir / f"valid.tsv", "w").writelines([root] + valid_tsv)
    open(dir / f"test.tsv", "w").writelines([root] + test_tsv)
    open(dir / f"train.km", "w").writelines(train_km)
    open(dir / f"valid.km", "w").writelines(valid_km)
    open(dir / f"test.km", "w").writelines(test_km)
    print(f"train: {len(train_km)}")
    print(f"valid: {len(valid_km)}")
    print(f"test: {len(test_km)}")
    print("done")
