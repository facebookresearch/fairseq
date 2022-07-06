# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import sys
from collections import defaultdict

ENG = "eng_Latn"


def get_type(pair):
    if "-" not in pair:
        return None
    from examples.nllb.modeling.evaluation.train_example_count import flores200_public

    train_counts2 = flores200_public.train_counts
    # 15M, 2-15M,0.1-2M, <0.1M
    low_limits = {"high": 1000000, "low": 0, "v_low": 0}
    high_limits = {"high": 1000000000, "low": 1000000, "v_low": 100000}
    lang = pair.split("-")[1]
    if lang == ENG:
        lang = pair.split("-")[0]
    if lang not in train_counts2:
        print(f"{lang} is not in train_counts")
        return None
    count = train_counts2[lang]
    for t in low_limits.keys():
        if count >= low_limits[t] and count <= high_limits[t]:
            return t


def get_averages(scores_map, threshold=50):
    en_xx = defaultdict(list)
    xx_en = defaultdict(list)
    non_eng = defaultdict(list)
    all_pairs = defaultdict(list)
    counts = defaultdict(int)
    for pair, score in scores_map.items():
        resource = get_type(pair)
        counts[resource] += 1
        if score > threshold:
            print(f"{pair} {score} is skipped due to threshold")
            continue
        if resource is None:
            print(f"{pair} {score} is skipped due to missing resource level")
            continue
        all_pairs["all"].append(score)
        all_pairs[resource].append(score)
        if pair.startswith(f"{ENG}-"):
            en_xx[resource].append(score)
            en_xx["all"].append(score)
        elif pair.endswith(f"-{ENG}"):
            xx_en[resource].append(score)
            xx_en["all"].append(score)
        else:
            non_eng[resource].append(score)
            non_eng["all"].append(score)
    print(counts)
    avg_en_xx = defaultdict(int)
    avg_xx_en = defaultdict(int)
    avg_non_eng = defaultdict(int)
    avg_all_pairs = defaultdict(int)
    lists = [en_xx, xx_en, non_eng, all_pairs]
    averages = [avg_en_xx, avg_xx_en, avg_non_eng, avg_all_pairs]
    for idx, agg in enumerate(averages):
        lst = lists[idx]
        for resource in ["all", "high", "low", "v_low"]:
            agg[resource] = round(sum(lst[resource]) / max(len(lst[resource]), 1), 2)
    return {
        "en-xx": avg_en_xx,
        "xx-en": avg_xx_en,
        "non-eng": avg_non_eng,
        "all": avg_all_pairs,
    }


def get_parser():
    parser = argparse.ArgumentParser(description="tabulates valid PPL")
    # fmt: off
    parser.add_argument(
        '--train-log',
        type=str,
        help='train log path'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=50,
        help='remove noisy values above this threshold'
    )
    parser.add_argument(
        '--updates',
        type=str,
        default="40000,60000,100000",
        help=''
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='output file')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    train_log = args.train_log
    update_values = args.updates.split(",")
    update_values = [int(v) for v in update_values]
    valid_ppls = defaultdict()
    for updates in update_values:
        valid_ppls[updates] = defaultdict()
    with open(train_log, "r") as f:
        for line in f.readlines():
            if "valid_" in line and "wps" in line:
                val = json.loads(line.split("|")[-1])
                ppl = 0.0
                updates = 0
                pair = ""
                for k, v in val.items():
                    if k.endswith("_ppl") and not k.endswith("_best_ppl"):
                        ppl = round(float(v), 2)
                        pair = k.replace("valid_main:", "").replace("_ppl", "")
                    elif "num_updates" in k:
                        updates = int(v)
                if updates in update_values and pair != "valid":
                    valid_ppls[updates][pair] = ppl

    pairs = valid_ppls[update_values[0]].keys()
    en_xx_pairs = [p for p in pairs if p.startswith(f"{ENG}-")]
    xx_en_pairs = [p for p in pairs if p.endswith(f"-{ENG}")]
    non_en_pairs = [p for p in pairs if "eng" not in p]
    pairs = en_xx_pairs + non_en_pairs + xx_en_pairs
    rows = defaultdict(list)
    for updates in update_values:
        average_vals = get_averages(valid_ppls[updates], args.threshold)
        for subset, avgs in average_vals.items():
            for res, val in avgs.items():
                rows[f"{subset}_{res}"].append(str(val))
        for pair in pairs:
            rows[pair].append(str(valid_ppls[updates][pair]))
    all_updates = "\t".join([str(v) for v in update_values])
    print(f"valid_ppl\t{all_updates}")
    if args.output_file is not None:
        fout = open(args.output_file, "w")
    else:
        fout = sys.stdout
    for k, v in rows.items():
        val = "\t".join(v)
        print(f"{k}\t{val}", file=fout)
    fout.close()


if __name__ == "__main__":
    main()
