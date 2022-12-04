#!/usr/bin/env python
import json
import os
import re
from glob import glob
from itertools import takewhile
from pathlib import Path
from typing import Dict, List

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


def find_last_matching_line(reversed_lns: List[str], pattern: str) -> Dict[str, str]:
    """Find a line with train loss in it and try to read it to json."""
    matched_line = None
    for l in reversed_lns:
        if pattern in l and "epoch" in l:
            matched_line = l
            break
    if matched_line is None:
        raise ValueError(f"none of lines had the substring {pattern}")

    if "{" in matched_line:  # log_format == 'json':
        strang = matched_line.split("|")[-1]
        record = json.loads(strang)
    elif pattern == "train_inner":
        strang = matched_line.split("|")[-1]
        record = parse_train_inner_record(strang)
    else:
        record = parse_pipe_separated_record(matched_line)
    epoch = record.pop("epoch", None)
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


def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split("\n")
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first
                if buffer[-1] != "\n":
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment


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
        lns = reverse_readline(f)
        try:
            record = find_last_matching_line(lns, pattern=log_pattern)
        except ValueError as e:
            print(f"failed to parse {f} with {str(e)}")
            continue
        record["parent_path"] = Path(f).parent.name
        record["fname"] = Path(f).name
        if f.startswith("multirun/"):  # produced by hydra
            _, date, t, *__ = f.split("/")
            record["date"] = f"{date}-{t}"
        records.append(record)

    if len(records) == 0:
        raise ValueError(
            f"None of the {len(matches)} log files are ready to be parsed."
        )
    df = pd.DataFrame(records)
    # Use the more informative path column. For sweep output this is parent path.
    # For manual log files it's usually fname
    path_col = (
        "parent_path"
        if (df["parent_path"].nunique() > df["fname"].nunique())
        else "fname"
    )
    df = df.set_index(path_col).pipe(tryfloat).round(2).sort_index()
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
