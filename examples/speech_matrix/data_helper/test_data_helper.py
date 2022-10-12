import os
import pandas as pd
import csv


def read_aud_manifest(fn):
    """
    Input:
        fn: audio manifest with columns aud_path, n_frames
    Return:
        aud_ids, nframes, aud_paths
    """
    aud_ids = []
    nframes = []
    aud_paths = []
    with open(fn, "r") as fin:
        aud_root = fin.readline().strip()
        for line in fin:
            aud, nf = line.strip().split()
            aud_id = ".".join(aud.split(".")[:-1])
            aud_ids.append(aud_id)
            nframes.append(int(nf))
            aud_path = os.path.join(aud_root, aud)
            aud_paths.append(aud_path)
    return aud_ids, nframes, aud_paths


def gen_s2u_manifest(aud_ids, nframes, aud_paths, s2u_manifest_fn):
    """
    No ground truth units are given in test data
    """
    s2u_dir = os.path.dirname(s2u_manifest_fn)
    if not os.path.isdir(s2u_dir):
        os.makedirs(s2u_dir)
    # columns: id, src_audio, src_n_frames, tgt_audio, tgt_n_frames
    data_dict = {
        "id": aud_ids,
        "src_audio": aud_paths,
        "src_n_frames": [int(nf / 160) for nf in nframes],
        "tgt_audio": [0 for idx in aud_ids],
        "tgt_n_frames": [1 for idx in aud_ids],
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


def gen_asr_manifest(aud_ids, asr_manifest_fn):
    """
    no ground truth units are given
    """
    asr_dir = os.path.dirname(asr_manifest_fn)
    if not os.path.isdir(asr_dir):
        os.makedirs(asr_dir)
    data_dict = {"id": aud_ids, "tgt_text": [0 for idx in aud_ids]}
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


def gen_manifest(aud_manifest_fn, s2u_manifest_fn, asr_manifest_fn):
    aud_ids, nframes, aud_paths = read_aud_manifest(aud_manifest_fn)
    gen_s2u_manifest(aud_ids, nframes, aud_paths, s2u_manifest_fn)
    gen_asr_manifest(aud_ids, asr_manifest_fn)
