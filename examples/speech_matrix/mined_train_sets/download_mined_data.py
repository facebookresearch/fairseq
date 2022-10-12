import os
import argparse
from examples.speech_matrix.data_helper.data_cfg import (
    DOWNLOAD_HUB,
    VP_LANGS,
    audio_key,
    aligned_speech_key,
)


def download_audio(lang, save_root):
    save_dir = os.path.join(save_root, audio_key)
    aud_dl = f"{DOWNLOAD_HUB}/{audio_key}/{lang}_aud.zip"
    os.system(f"wget {aud_dl} -P {save_dir}")


def download_aligned_speech(src_lang, tgt_lang, save_root):
    save_dir = os.path.join(save_root, aligned_speech_key)
    s2s_dl = f"{DOWNLOAD_HUB}/{aligned_speech_key}/{src_lang}-{tgt_lang}.tsv.gz"
    os.system(f"wget {s2s_dl} -P {save_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root", type=str, required=True)
    args = parser.parse_args()

    # download audio zips
    for lang in VP_LANGS:
        download_audio(lang=lang, save_root=args.save_root)

    # download speech alignments
    for src_lang in VP_LANGS:
        for tgt_lang in VP_LANGS:
            if src_lang >= tgt_lang:
                continue
            download_aligned_speech(src_lang, tgt_lang, save_root=args.save_root)
