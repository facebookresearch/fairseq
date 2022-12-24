import os
import argparse
from examples.speech_matrix.data_helper.data_cfg import (
    DOWNLOAD_HUB,
    VP_LANGS,
    trans_key,
    TRANS_SRC_LANGS,
    TRANS_TGT_LANGS
)


def download_transcriptions(src_lang, tgt_lang, save_root):
    save_dir = os.path.join(save_root, trans_key)
    os.makedirs(save_dir, exist_ok=True)
    
    sorted_src_lang, sorted_tgt_lang = sorted([src_lang, tgt_lang])
    s2s_dl = f"{DOWNLOAD_HUB}/{trans_key}/{sorted_src_lang}-{sorted_tgt_lang}_{tgt_lang}.tsv.gz"
    os.system(f"wget {s2s_dl} -P {save_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root", type=str, required=True)
    args = parser.parse_args()

    # download transcriptions of ta
    for src_lang in TRANS_SRC_LANGS:
        for tgt_lang in TRANS_TGT_LANGS:
            if src_lang == tgt_lang:
                continue
            download_transcriptions(
                src_lang, tgt_lang, save_root=args.save_root
            )
