import os
import argparse
from pathlib import Path
import pandas as pd
import csv
import yaml
from examples.speech_matrix.data_helper.data_cfg import (
    VP_LANGS,
    high_res_pairs,
    mid_res_pairs,
    manifest_key,
    audio_key,
    manifest_prefix,
)
from examples.speech_matrix.data_helper.model_cfg import hubert_config_hub


def get_mine_threshold(lang_pair):
    sorted_lang_pair = "-".join(sorted(lang_pair.split("-")))
    if sorted_lang_pair in high_res_pairs:
        return 1.09
    elif sorted_lang_pair in mid_res_pairs:
        return 1.07
    else:
        return 1.06


def parse_out_fn(fn, td):
    in_dir = os.path.dirname(fn)
    in_fn = os.path.basename(fn)
    out_fn = ".".join(in_fn.split(".")[:-1]) + f"_t{td}.tsv"
    return os.path.join(in_dir, out_fn)


def filter_dataframe(df, td):
    res_df = df.loc[df["score"] >= td]
    return res_df


def prep_data_with_threshold(lang_pair, save_root):
    manifest_dir = os.path.join(save_root, manifest_key, lang_pair)
    s2u_manifest = os.path.join(manifest_dir, f"{manifest_prefix}.tsv")
    asr_manifest = os.path.join(manifest_dir, "source_unit", f"{manifest_prefix}.tsv")

    td = get_mine_threshold(lang_pair)
    out_s2u_manifest = parse_out_fn(s2u_manifest, td)
    out_asr_manifest = parse_out_fn(asr_manifest, td)
    s2u_df = pd.read_csv(
        s2u_manifest,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )
    s2u_df = filter_dataframe(s2u_df, td)
    s2u_df.to_csv(out_s2u_manifest, sep="\t", index=False)
    print("save to {}".format(out_s2u_manifest))

    asr_df = pd.read_csv(
        asr_manifest,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )
    asr_df["score"] = s2u_df["score"]
    asr_df = filter_dataframe(asr_df, td)
    asr_df.drop(columns=["score"], axis=1)
    asr_df.to_csv(out_asr_manifest, sep="\t", index=False)
    print("save to {}".format(out_asr_manifest))


def gen_config(src_lang, tgt_lang, save_root):
    """
    config.yaml
    config_multitask.yaml
    """
    ref_config_dir = os.path.join(Path(__file__).parent.parent, "data_helper")
    ref_config = os.path.join(ref_config_dir, "config.yaml")
    ref_config_multitask = os.path.join(ref_config_dir, "config_multitask.yaml")

    s2u_dir = os.path.join(save_root, manifest_key, f"{src_lang}-{tgt_lang}")
    config = os.path.join(s2u_dir, "config.yaml")
    config_multitask = os.path.join(s2u_dir, "config_multitask.yaml")
    asr_dir = os.path.join(s2u_dir, "source_unit")

    # update config
    with open(ref_config, "r") as fin:
        data = yaml.safe_load(fin)
    data["audio_root"] = os.path.join(save_root, audio_key)
    with open(config, "w") as fout:
        yaml.dump(data, fout)
    print(f"{config} prepared.")

    # update config_multitask
    with open(ref_config_multitask, "r") as fin:
        data = yaml.safe_load(fin)
    data["source_unit"]["data"] = asr_dir
    data["source_unit"]["dict"] = os.path.join(asr_dir, "dict.txt")
    with open(config_multitask, "w") as fout:
        yaml.dump(data, fout)
    print(f"{config_multitask} prepared.")


def gen_dict(src_lang, tgt_lang, save_root):
    """
    dictionary for source units
    """
    _, _, km = hubert_config_hub[src_lang]
    s2u_dir = os.path.join(save_root, manifest_key, f"{src_lang}-{tgt_lang}")
    asr_dir = os.path.join(s2u_dir, "source_unit")
    dict_fn = os.path.join(asr_dir, "dict.txt")
    with open(dict_fn, "w") as fout:
        for idx in range(km):
            fout.write(f"{idx} 1\n")
    print("dict prepared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root", type=str, required=True)
    args = parser.parse_args()

    for src_lang in VP_LANGS:
        for tgt_lang in VP_LANGS:
            if src_lang == tgt_lang:
                continue
            prep_data_with_threshold(
                lang_pair=f"{src_lang}-{tgt_lang}", save_root=args.save_root
            )
            # config
            gen_config(src_lang, tgt_lang, save_root=args.save_root)
            # dict
            gen_dict(src_lang, tgt_lang, save_root=args.save_root)
