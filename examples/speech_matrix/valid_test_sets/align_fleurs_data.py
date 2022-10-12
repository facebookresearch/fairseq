import os
import argparse
from collections import defaultdict
from examples.speech_matrix.data_helper.data_cfg import (
    FLEURS_LANGS,
    FLORES_LANG_MAP,
    manifest_key,
)
from examples.speech_matrix.data_helper.cleaners import text_cleaners


domain = "fleurs"


def read_flores_texts(flores_root, flores_lang, splits=["dev", "devtest"]):
    texts = []
    for split in splits:
        fn = os.path.join(flores_root, split, f"{flores_lang}.{split}")
        with open(fn, "r") as fin:
            texts += [line.strip() for line in fin.readlines()]
    return texts


def read_aud_manifest(aud_manifest):
    aud_list = []
    with open(aud_manifest, "r") as fin:
        aud_root = fin.readline().strip()
        for line in fin:
            aud_list.append(line.strip())
    return aud_root, aud_list


def read_raw_trans_fn(raw_trans_fn):
    raw_texts = []
    with open(raw_trans_fn, "r") as fin:
        for line in fin:
            raw_texts.append(line.strip())
    return raw_texts


def map_raw_text_to_speech(aud_manifest, trans_fn, raw_trans_fn):
    """
    raw_texts are ordered
    using raw_text as key to find corresponding speech
    """
    aud_root, aud_list = read_aud_manifest(aud_manifest)
    texts = read_raw_trans_fn(trans_fn)
    raw_texts = read_raw_trans_fn(raw_trans_fn)
    raw_text_to_speech = defaultdict(list)
    raw_text_to_text = defaultdict(str)
    for raw_text, aud, text in zip(raw_texts, aud_list, texts):
        raw_text_to_speech[raw_text].append(aud)
        raw_text_to_text[raw_text] = text
    return aud_root, raw_text_to_speech, raw_text_to_text


def align_speech_to_speech(
    # input args
    flores_root,
    src_lang,
    tgt_lang,
    src_flores_lang,
    tgt_flores_lang,
    src_aud_manifest,
    src_trans_fn,
    src_raw_trans_fn,
    tgt_aud_manifest,
    tgt_trans_fn,
    tgt_raw_trans_fn,
    # output s2t
    s2t_src_aud_manifest,
    s2t_tgt_aud_manifest,
    s2t_trans_fn,
    # output t2s
    t2s_tgt_aud_manifest,
    t2s_src_aud_manifest,
    t2s_trans_fn,
):
    """
    (sa1, sa2, sa3) <-> st (src audios <-> flores text)
    (ta1, ta2) <-> tt (tgt audios <-> flores text)
    s2t_src_aud/trans: sa1 -> tt, sa2 -> tt, sa3 -> tt
    s2t_tgt_aud: ta1, ta2, ta1 (to match the sample size of src_src_aud)

    t2s_tgt_aud/trans: ta1 -> st, ta2 -> st
    t2s_src_aud/trans: sa1, sa2
    """

    src_aud_root, src_raw_text_to_speech, src_raw_text_to_text = map_raw_text_to_speech(
        src_aud_manifest, src_trans_fn, src_raw_trans_fn
    )
    tgt_aud_root, tgt_raw_text_to_speech, tgt_raw_text_to_text = map_raw_text_to_speech(
        tgt_aud_manifest, tgt_trans_fn, tgt_raw_trans_fn
    )

    # output
    s2t_src_aud_fout = open(s2t_src_aud_manifest, "w")
    s2t_src_aud_fout.write(src_aud_root + "\n")
    s2t_tgt_aud_fout = open(s2t_tgt_aud_manifest, "w")
    s2t_tgt_aud_fout.write(tgt_aud_root + "\n")
    s2t_trans_fout = open(s2t_trans_fn, "w")

    t2s_tgt_aud_fout = open(t2s_tgt_aud_manifest, "w")
    t2s_tgt_aud_fout.write(tgt_aud_root + "\n")
    t2s_src_aud_fout = open(t2s_src_aud_manifest, "w")
    t2s_src_aud_fout.write(src_aud_root + "\n")
    t2s_trans_fout = open(t2s_trans_fn, "w")

    # raw text alignment in flores
    src_flores_texts = read_flores_texts(flores_root, src_flores_lang)
    tgt_flores_texts = read_flores_texts(flores_root, tgt_flores_lang)

    for src_raw_text, tgt_raw_text in zip(src_flores_texts, tgt_flores_texts):
        src_aud_list = src_raw_text_to_speech[src_raw_text]
        src_text = src_raw_text_to_text[src_raw_text]

        tgt_aud_list = tgt_raw_text_to_speech[tgt_raw_text]
        tgt_text = tgt_raw_text_to_text[tgt_raw_text]

        # src_lang -> tgt_lang
        for src_idx, src_aud in enumerate(src_aud_list):
            if len(tgt_aud_list) != 0:
                # src_aud
                s2t_src_aud_fout.write(src_aud + "\n")
                s2t_trans_fout.write(text_cleaners(tgt_text, tgt_lang) + "\n")
                # tgt_aud
                tgt_idx = src_idx % len(tgt_aud_list)
                s2t_tgt_aud_fout.write(tgt_aud_list[tgt_idx] + "\n")

        # tgt_lang -> src_lang
        for tgt_idx, tgt_aud in enumerate(tgt_aud_list):
            if len(src_aud_list) != 0:
                # tgt_aud
                t2s_tgt_aud_fout.write(tgt_aud + "\n")
                t2s_trans_fout.write(text_cleaners(src_text, src_lang) + "\n")
                # src_aud
                src_idx = tgt_idx % len(src_aud_list)
                t2s_src_aud_fout.write(src_aud_list[src_idx] + "\n")

    s2t_src_aud_fout.close()
    s2t_tgt_aud_fout.close()
    s2t_trans_fout.close()

    t2s_tgt_aud_fout.close()
    t2s_src_aud_fout.close()
    t2s_trans_fout.close()
    print(f"save to {s2t_src_aud_manifest}")
    print(f"save to {t2s_tgt_aud_manifest}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Aligning FLEURS and FLORES")
    parser.add_argument("--flores-root", type=str, required=True)
    parser.add_argument("--proc-fleurs-dir", type=str, required=True)
    parser.add_argument("--save-root", type=str, required=True)
    args = parser.parse_args()

    manifest_root = os.path.join(args.save_root, manifest_key)
    splits = ["valid", "test"]
    fleurs_lang_num = len(FLEURS_LANGS)
    fleurs_manifest_dir = os.path.join(args.proc_fleurs_dir, "aud_manifests")

    for src_idx in range(fleurs_lang_num):
        for tgt_idx in range(src_idx + 1, fleurs_lang_num):
            src_fleurs_lang = FLEURS_LANGS[src_idx]
            src_lang = src_fleurs_lang[:2]
            src_flores_lang = FLORES_LANG_MAP[src_lang]

            tgt_fleurs_lang = FLEURS_LANGS[tgt_idx]
            tgt_lang = tgt_fleurs_lang[:2]
            tgt_flores_lang = FLORES_LANG_MAP[tgt_lang]

            # src_lang -> tgt_lang
            src_to_tgt_dir = os.path.join(fleurs_manifest_dir, f"{src_lang}-{tgt_lang}")
            os.makedirs(src_to_tgt_dir, exist_ok=True)
            # tgt_lang -> src_lang
            tgt_to_src_dir = os.path.join(fleurs_manifest_dir, f"{tgt_lang}-{src_lang}")
            os.makedirs(tgt_to_src_dir, exist_ok=True)
            s2t_manifest_dir = os.path.join(manifest_root, f"{src_lang}-{tgt_lang}")
            os.makedirs(s2t_manifest_dir, exist_ok=True)
            t2s_manifest_dir = os.path.join(manifest_root, f"{tgt_lang}-{src_lang}")
            os.makedirs(t2s_manifest_dir, exist_ok=True)

            for split in splits:
                print(f"{split}: {src_lang}<>{tgt_lang}")
                align_speech_to_speech(
                    # input args
                    flores_root=args.flores_root,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    src_flores_lang=src_flores_lang,
                    tgt_flores_lang=tgt_flores_lang,
                    src_aud_manifest=os.path.join(
                        fleurs_manifest_dir, f"{split}_{src_lang}.tsv"
                    ),
                    src_trans_fn=os.path.join(
                        fleurs_manifest_dir, f"{split}_{src_lang}.trans"
                    ),
                    src_raw_trans_fn=os.path.join(
                        fleurs_manifest_dir, f"{split}_{src_lang}.raw.trans"
                    ),
                    tgt_aud_manifest=os.path.join(
                        fleurs_manifest_dir, f"{split}_{tgt_lang}.tsv"
                    ),
                    tgt_trans_fn=os.path.join(
                        fleurs_manifest_dir, f"{split}_{tgt_lang}.trans"
                    ),
                    tgt_raw_trans_fn=os.path.join(
                        fleurs_manifest_dir, f"{split}_{tgt_lang}.raw.trans"
                    ),
                    # output args
                    s2t_src_aud_manifest=os.path.join(
                        src_to_tgt_dir, f"{split}_{src_lang}-{tgt_lang}_{src_lang}.tsv"
                    ),
                    s2t_tgt_aud_manifest=os.path.join(
                        src_to_tgt_dir, f"{split}_{src_lang}-{tgt_lang}_{tgt_lang}.tsv"
                    ),
                    s2t_trans_fn=os.path.join(
                        s2t_manifest_dir, f"{split}_{domain}.{tgt_lang}"
                    ),
                    # t2s
                    t2s_tgt_aud_manifest=os.path.join(
                        tgt_to_src_dir, f"{split}_{tgt_lang}-{src_lang}_{tgt_lang}.tsv"
                    ),
                    t2s_src_aud_manifest=os.path.join(
                        tgt_to_src_dir, f"{split}_{tgt_lang}-{src_lang}_{src_lang}.tsv"
                    ),
                    t2s_trans_fn=os.path.join(
                        t2s_manifest_dir, f"{split}_{domain}.{src_lang}"
                    ),
                )
