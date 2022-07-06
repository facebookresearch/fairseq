# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Used to extract metrics from train.log (pattern=valid_main for per-language). Example:
> python examples/nllb/modeling/scripts/get_train_log_metrics.py --filepath $FILEPATH --pattern \
    valid_main --metric ppl --print-steps --src eng --tgt
"""


import argparse
import ast
import subprocess


def get_train_log_metrics(
    file,
    pattern="valid_main",
    metric="ppl",
    print_steps=True,
    interactive=False,
    src="eng",
    tgt=None,
):
    wildcard = "[a-z]+(_[A-Z])?[a-z]+"
    direction = f"{src or wildcard}-{tgt or wildcard}"
    if pattern == "valid_main":
        p1 = subprocess.Popen(
            ["grep", "-E", "--text", f"\| valid_main:{direction} \|", file],
            stdout=subprocess.PIPE,
        )
    else:
        p1 = subprocess.Popen(
            ["grep", "--text", f"| {pattern} |", file], stdout=subprocess.PIPE
        )
    p2 = subprocess.Popen(
        ["cut", "-d", "|", "-f", "4"], stdin=p1.stdout, stdout=subprocess.PIPE
    )
    if pattern == "valid_main":
        p3 = subprocess.Popen(
            [
                "jq",
                "-r",
                "-c",
                f'with_entries( select( (.key | match("valid_main:{direction}(_num_updates|_{metric})")) ) )',
            ],
            stdin=p2.stdout,
            stdout=subprocess.PIPE,
        )
    elif pattern == "train_inner":
        p3 = subprocess.Popen(
            [
                "jq",
                "-r",
                "-c",
                f'with_entries( select( (.key | match("num_updates|{metric}")) ) )',
            ],
            stdin=p2.stdout,
            stdout=subprocess.PIPE,
        )
    else:
        p3 = subprocess.Popen(
            [
                "jq",
                "-r",
                "-c",
                f'with_entries( select( (.key | match("{pattern}_num_updates|{pattern}_{metric}")) ) )',
            ],
            stdin=p2.stdout,
            stdout=subprocess.PIPE,
        )
    p1.stdout.close()
    p2.stdout.close()
    all_p3 = p3.communicate()[0].decode("utf-8").strip().split("\n")

    if pattern == "valid_main":
        languages_results = {}
        for entry in all_p3:
            d = ast.literal_eval(entry)
            for k, v in d.items():
                if metric in k:
                    lang_pair = k.replace("valid_main:", "").replace(f"_{metric}", "")
                    if k not in languages_results:
                        languages_results[k] = {}
                    num_steps = d["valid_main:" + lang_pair + "_num_updates"]
                    languages_results[k][num_steps] = v

        if interactive:
            return languages_results
        for k, v in languages_results.items():
            if print_steps:
                print(f"{k}\t{v}")
            else:
                print(f"{k}\t{v.values()}")
    else:
        results = {}
        key_types = []
        entry = all_p3[0]
        d = ast.literal_eval(entry)
        for k, v in d.items():
            key_type = k.removeprefix(f"{pattern}_")
            if key_type != "num_updates":
                key_types.append(key_type)
                results[key_type] = {}
        for entry in all_p3:
            d = ast.literal_eval(entry)
            for key_type in key_types:
                if pattern == "train_inner":
                    num_steps = d["num_updates"]
                    results[key_type][num_steps] = d[key_type]
                else:
                    num_steps = d[f"{pattern}_num_updates"]
                    results[key_type][num_steps] = d[f"{pattern}_{key_type}"]
        if interactive:
            return results
        for key_type, res in results.items():
            print(key_type)
            if print_steps:
                print(res)
            else:
                print(res.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="path to train.log")
    parser.add_argument("--pattern", type=str, default="valid_main")
    parser.add_argument("--metric", type=str, default="ppl")
    parser.add_argument("--print-steps", action="store_true")
    parser.add_argument("--src", nargs="?", default="eng")
    parser.add_argument("--tgt", nargs="?", default=None)
    args = parser.parse_args()
    get_train_log_metrics(
        args.filepath,
        args.pattern,
        args.metric,
        args.print_steps,
        src=args.src,
        tgt=args.tgt,
    )
