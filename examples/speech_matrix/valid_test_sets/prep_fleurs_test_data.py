import os
import argparse
from examples.speech_matrix.data_helper.data_cfg import FLEURS_LANGS, manifest_key
from examples.speech_matrix.data_helper.test_data_helper import gen_manifest


domain = "fleurs"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--proc-fleurs-dir", type=str, required=True)
    parser.add_argument("--save-root", type=str, required=True)
    args = parser.parse_args()

    fleurs_manifest_dir = os.path.join(args.proc_fleurs_dir, "aud_manifests")
    manifest_dir = os.path.join(args.save_root, manifest_key)
    lang_num = len(FLEURS_LANGS)

    for src_idx in range(lang_num):
        for tgt_idx in range(src_idx + 1, lang_num):
            src_lang = FLEURS_LANGS[src_idx]
            tgt_lang = FLEURS_LANGS[tgt_idx]
            src_lang_code = src_lang[:2]
            tgt_lang_code = tgt_lang[:2]

            # src_lang -> tgt_lang
            print(f"processing {src_lang_code}-{tgt_lang_code}...")
            src_aud_manifest = os.path.join(
                fleurs_manifest_dir,
                f"{src_lang_code}-{tgt_lang_code}",
                f"test_{src_lang_code}-{tgt_lang_code}_{src_lang_code}.tsv",
            )
            src_s2u_manifest = os.path.join(
                manifest_dir, f"{src_lang_code}-{tgt_lang_code}", f"test_{domain}.tsv"
            )
            src_asr_manifest = os.path.join(
                manifest_dir,
                f"{src_lang_code}-{tgt_lang_code}",
                "source_unit",
                f"test_{domain}.tsv",
            )
            gen_manifest(src_aud_manifest, src_s2u_manifest, src_asr_manifest)

            # tgt_lang -> src_lang
            print(f"processing {tgt_lang_code}-{src_lang_code}...")
            tgt_aud_manifest = os.path.join(
                fleurs_manifest_dir,
                f"{tgt_lang_code}-{src_lang_code}",
                f"test_{tgt_lang_code}-{src_lang_code}_{tgt_lang_code}.tsv",
            )
            tgt_s2u_manifest = os.path.join(
                manifest_dir, f"{tgt_lang_code}-{src_lang_code}", f"test_{domain}.tsv"
            )
            tgt_asr_manifest = os.path.join(
                manifest_dir,
                f"{tgt_lang_code}-{src_lang_code}",
                "source_unit",
                f"test_{domain}.tsv",
            )
            gen_manifest(tgt_aud_manifest, tgt_s2u_manifest, tgt_asr_manifest)
