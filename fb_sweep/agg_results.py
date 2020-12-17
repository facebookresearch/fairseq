#!/usr/bin/env python
from pathlib import Path
from glob import glob

try:
    import pandas as pd
    from fire import Fire
except ImportError:
    "results aggregation has extra dependencies. Run `pip install pandas fire tabulate`."
import json


def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def find_last_matching_line(path: str, pattern: str = "train_loss") -> dict[str, float]:
    """ Find a line with train loss in it and try to read it to json."""
    lns = Path(path).open().read().split("\n")
    matched_line = None
    for l in reversed(lns):
        if pattern in l:
            matched_line = l
            break
    if matched_line is None:
        raise ValueError(f"none of {len(lns)} lines had the substring train_loss")
    strang = matched_line.split("|")[-1]
    record = json.loads(strang)
    return record


def find_all_matching_lines(path, pattern="train_inner"):
    """Read train_inner logs (each step)"""
    lns = Path(path).open().read().split("\n")
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
    matches = list(glob(pattern))
    for f in matches:
        try:
            record = find_last_matching_line(f, pattern=log_pattern)
            record["path"] = Path(f).parent.name
            records.append(record)
        except Exception as e:
            print(f"Failed on {f} with {e}")
    if len(records) == 0:
        raise ValueError(
            f"None of the {len(matches)} log files are ready to be parsed."
        )
    df = pd.DataFrame(records)
    df = df.set_index("path").pipe(tryfloat).round(2).sort_index()
    df = df.rename(columns=lambda x: remove_prefix(x, "train_").replace(".", "_"))
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
