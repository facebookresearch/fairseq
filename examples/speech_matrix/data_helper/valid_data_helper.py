import os
import pandas as pd
import csv


def parse_unit_fn(unit_fn):
    units = []
    with open(unit_fn, "r") as fin:
        for line in fin:
            line = line.strip()
            units.append(line)
    return units


def gen_s2u_manifest(aud_ids, nframes, aud_paths, units, s2u_manifest_fn):
    """
    ground truth units are given
    """
    assert len(aud_ids) == len(units)
    reduced_units = [dedup_unit(unit) for unit in units]
    data_dict = {
        "id": aud_ids,
        "src_audio": aud_paths,
        "src_n_frames": [int(nf / 160) for nf in nframes],
        "tgt_audio": reduced_units,
        "tgt_n_frames": [len(unit.split()) for unit in reduced_units],
    }
    s2u_df = pd.DataFrame(data_dict)
    s2u_df.to_csv(
        s2u_manifest_fn,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )
    print("save to {}".format(s2u_manifest_fn))


def dedup_unit(unit):
    re_unit = []
    prev = None
    for val in unit.split():
        if val == prev:
            continue
        else:
            re_unit.append(val)
            prev = val
    return " ".join(re_unit)


def gen_asr_manifest(aud_ids, units, asr_manifest_fn):
    """
    ground truth units are given
    """
    assert len(aud_ids) == len(units)
    asr_dir = os.path.dirname(asr_manifest_fn)
    os.makedirs(asr_dir, exist_ok=True)
    data_dict = {"id": aud_ids, "tgt_text": [dedup_unit(unit) for unit in units]}
    asr_df = pd.DataFrame(data_dict)
    asr_df.to_csv(
        asr_manifest_fn,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )
    print("save to {}".format(asr_manifest_fn))
