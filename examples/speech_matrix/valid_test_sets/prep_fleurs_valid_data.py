import os
import argparse
from examples.speech_matrix.data_helper.data_cfg import FLEURS_LANGS, manifest_key
from examples.speech_matrix.data_helper.test_data_helper import read_aud_manifest
from examples.speech_matrix.data_helper.valid_data_helper import (
    gen_s2u_manifest,
    gen_asr_manifest,
    parse_unit_fn,
)
from examples.speech_matrix.data_helper.hubert_helper import extract_lang_units


domain = "fleurs"


def gen_valid_data_manifest(
    src_aud_manifest, tgt_unit_fn, src_unit_fn, s2u_manifest, asr_manifest
):
    aud_ids, nframes, aud_paths = read_aud_manifest(src_aud_manifest)
    tgt_units = parse_unit_fn(tgt_unit_fn)
    src_units = parse_unit_fn(src_unit_fn)
    assert len(aud_ids) == len(tgt_units)
    assert len(aud_ids) == len(src_units)
    gen_s2u_manifest(aud_ids, nframes, aud_paths, tgt_units, s2u_manifest)
    gen_asr_manifest(aud_ids, src_units, asr_manifest)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("FLEURS valid set preparation")
    parser.add_argument("--proc-fleurs-dir", type=str, required=True)
    parser.add_argument("--save-root", type=str, required=True)
    parser.add_argument("--hubert-model-dir", type=str, required=True)
    args = parser.parse_args()

    manifest_root = os.path.join(args.save_root, manifest_key)
    fleurs_manifest_dir = os.path.join(args.proc_fleurs_dir, "aud_manifests")

    for src_lang in FLEURS_LANGS:
        src_lang_code = src_lang[:2]
        for tgt_lang in FLEURS_LANGS:
            if src_lang == tgt_lang:
                continue
            tgt_lang_code = tgt_lang[:2]

            print(f"processing {src_lang_code}-{tgt_lang_code}...")
            manifest_dir = os.path.join(
                fleurs_manifest_dir,
                f"{src_lang_code}-{tgt_lang_code}",
            )
            src_split = f"valid_{src_lang_code}-{tgt_lang_code}_{src_lang_code}"
            src_aud_manifest = os.path.join(manifest_dir, f"{src_split}.tsv")
            extract_lang_units(
                src_aud_manifest,
                src_lang_code,
                manifest_dir,
                args.hubert_model_dir,
            )
            src_unit_fn = os.path.join(manifest_dir, f"{src_split}.km")

            tgt_split = f"valid_{src_lang_code}-{tgt_lang_code}_{tgt_lang_code}"
            tgt_aud_manifest = os.path.join(manifest_dir, f"{tgt_split}.tsv")
            extract_lang_units(
                tgt_aud_manifest,
                tgt_lang_code,
                manifest_dir,
                args.hubert_model_dir,
            )
            tgt_unit_fn = os.path.join(manifest_dir, f"{tgt_split}.km")

            s2u_manifest = os.path.join(
                manifest_root, f"{src_lang_code}-{tgt_lang_code}", f"valid_{domain}.tsv"
            )
            asr_manifest = os.path.join(
                manifest_root,
                f"{src_lang_code}-{tgt_lang_code}",
                "source_unit",
                f"valid_{domain}.tsv",
            )
            os.makedirs(
                os.path.join(
                    manifest_root, f"{src_lang_code}-{tgt_lang_code}", "source_unit"
                ),
                exist_ok=True,
            )
            gen_valid_data_manifest(
                src_aud_manifest, tgt_unit_fn, src_unit_fn, s2u_manifest, asr_manifest
            )
