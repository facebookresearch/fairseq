#!/usr/bin/env python
from itertools import takewhile
from glob import glob
import json
from pathlib import Path
import re
from typing import List, Dict

try:
    import pandas as pd
    from fire import Fire
except ImportError:
    raise ImportError(
        "results aggregation has extra dependencies. Run `pip install pandas fire tabulate`."
    )


def find_common_prefix(parsed_keys: List[str]) -> str:
    # Finding common prefix using itertools.takewhile
    if len(parsed_keys) <= 1:
        return ""
    return "".join(
        c[0] for c in takewhile(lambda x: all(x[0] == y for y in x), zip(*parsed_keys))
    )


def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def remove_common_prefix_from_keys(entry: Dict[str, str]):
    common_prefix = find_common_prefix(entry.keys())
    if not common_prefix:
        return entry
    else:
        return {remove_prefix(k, common_prefix): v for k, v in entry.items()}


def find_last_matching_line(lns: List[str], pattern: str) -> Dict[str, str]:
    """ Find a line with train loss in it and try to read it to json."""
    matched_line = None
    for l in reversed(lns):
        if pattern in l and "epoch" in l:
            matched_line = l
            break
    if matched_line is None:
        raise ValueError(f"none of {len(lns)} lines had the substring {pattern}")

    if "{" in matched_line:  # log_format == 'json':
        strang = matched_line.split("|")[-1]
        record = json.loads(strang)
    elif pattern == "train_inner":
        strang = matched_line.split("|")[-1]
        record = parse_train_inner_record(strang)
    else:
        record = parse_pipe_separated_record(matched_line)
    epoch = record.pop("epoch")
    sanitized_record = remove_common_prefix_from_keys(record)
    sanitized_record["epoch"] = epoch
    return sanitized_record


def parse_pipe_separated_record(rec2):
    pipe_sep = [
        x.strip()
        for x in rec2.split("INFO")[-1].split("|")
        if x.strip().count(" ") == 1
    ]
    return dict([entry.split(" ") for entry in pipe_sep])


def parse_train_inner_record(record):
    kv_pairs: str = re.compile(r"(\/ \d+\s)(.+)").search(record).groups()[-1]
    return dict([entry.strip().split("=") for entry in kv_pairs.split(",")])


def find_all_matching_lines(lns, pattern="train_inner"):
    """Read train_inner logs (each step)"""
    records = []
    for l in lns:
        if pattern not in l:
            continue
        strang = l.split("|")[-1]
        record = json.loads(strang)
        records.append(record)
    return pd.DataFrame(records).pipe(tryfloat)


def tryfloat(x):
    if isinstance(x, pd.Series):
        try:
            return x.astype(float)
        except Exception:
            return x
    elif isinstance(x, pd.DataFrame):
        return x.apply(tryfloat)
    else:
        try:
            return float(x)
        except TypeError:
            return x


def make_sweep_table(
    pattern,
    log_pattern="train_inner",
    csv_path=None,
    keep_cols=None,
    sort_col=None,
    interactive=False,
):
    """
    For each file matching pattern, extract the last json line matching log_pattern, tabulate.
    Args:
        pattern: (str) files to consider, e.g.
            /checkpoint/sshleifer/2020-11-23/*/train.log* (should be quoted on command line)
        log_pattern: (str): usually train, train_inner or valid
        csv_path: (str) where to save if suffix is .md will save markdown, otherwise csv.
        keep_cols: (list) (comma separated from CL) column names to show
        sort_col: (str) column to sort resulting table by
        interactive: (bool) just return the DataFrame
    Usage:
        ./fb_sweep/agg_results.py "/checkpoint/sshleifer/2020-12-1*/big_run*/train.log" \
            --log-pattern valid \
            --csv_path big_run_sweep_results \
            --keep_cols train_ppl,train_wps,train_loss \
            --sort_col train_wps
    """
    records = []
    matches = list(glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"found no files matching {pattern}")
    for f in matches:
        try:
            lns = Path(f).open().read().split("\n")
            record = find_last_matching_line(lns, pattern=log_pattern)
            record["path"] = Path(f).parent.name
            if f.startswith("multirun/"):  # produced by hydra
                _, date, t, *__ = f.split("/")
                record["date"] = f"{date}-{t}"
            records.append(record)
        except Exception as e:
            print(f"Failed on {f} with {e}")
    if len(records) == 0:
        raise ValueError(
            f"None of the {len(matches)} log files are ready to be parsed."
        )
    df = pd.DataFrame(records)
    df = df.set_index("path").pipe(tryfloat).round(2).sort_index()
    df = df.rename(columns=lambda x: x.replace(".", "_"))
    if keep_cols is not None:
        df = df[list(keep_cols)]
    if sort_col is not None:
        df = df.sort_values(sort_col)
    if interactive:
        return df
    if csv_path is not None:
        if csv_path.endswith("md"):
            df.to_markdown(Path(csv_path).open("w"))
        else:
            df.to_csv(csv_path)
    print(df.to_markdown(tablefmt="grid"))


if __name__ == "__main__":
    # Usage: see docstring of make_sweep_table
    Fire(make_sweep_table)
