# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
import sys
from collections import defaultdict

# Ablation dataset only
datalabels = [
    "autshumato",
    "wmt",
    "wat",
    "tico",
    "pmindia",
    "iwslt",
    "floresv1",
    "lafand",
]

lang_pairs = [
    "afr_Latn-eng_Latn",
    "als_Latn-eng_Latn",
    "amh_Ethi-eng_Latn",
    "arb_Arab-eng_Latn",
    "azj_Latn-eng_Latn",
    "bam_Latn-fra_Latn",
    "bel_Cyrl-eng_Latn",
    "ben_Beng-eng_Latn",
    "bos_Latn-eng_Latn",
    "bul_Cyrl-eng_Latn",
    "ces_Latn-eng_Latn",
    "ckb_Arab-eng_Latn",
    "dan_Latn-eng_Latn",
    "deu_Latn-eng_Latn",
    "ell_Grek-eng_Latn",
    "eng_Latn-afr_Latn",
    "eng_Latn-als_Latn",
    "eng_Latn-amh_Ethi",
    "eng_Latn-arb_Arab",
    "eng_Latn-azj_Latn",
    "eng_Latn-bel_Cyrl",
    "eng_Latn-ben_Beng",
    "eng_Latn-bos_Latn",
    "eng_Latn-bul_Cyrl",
    "eng_Latn-ces_Latn",
    "eng_Latn-ckb_Arab",
    "eng_Latn-dan_Latn",
    "eng_Latn-deu_Latn",
    "eng_Latn-ell_Grek",
    "eng_Latn-est_Latn",
    "eng_Latn-fin_Latn",
    "eng_Latn-fra_Latn",
    "eng_Latn-fuv_Latn",
    "eng_Latn-gaz_Latn",
    "eng_Latn-glg_Latn",
    "eng_Latn-guj_Gujr",
    "eng_Latn-hau_Latn",
    "eng_Latn-heb_Hebr",
    "eng_Latn-hin_Deva",
    "eng_Latn-hrv_Latn",
    "eng_Latn-hun_Latn",
    "eng_Latn-hye_Armn",
    "eng_Latn-ibo_Latn",
    "eng_Latn-ind_Latn",
    "eng_Latn-ita_Latn",
    "eng_Latn-jpn_Jpan",
    "eng_Latn-kat_Geor",
    "eng_Latn-kaz_Cyrl",
    "eng_Latn-khk_Cyrl",
    "eng_Latn-khm_Khmr",
    "eng_Latn-kmr_Latn",
    "eng_Latn-kor_Hang",
    "eng_Latn-lin_Latn",
    "eng_Latn-lit_Latn",
    "eng_Latn-lug_Latn",
    "eng_Latn-luo_Latn",
    "eng_Latn-lvs_Latn",
    "eng_Latn-mal_Mlym",
    "eng_Latn-mar_Deva",
    "eng_Latn-mkd_Cyrl",
    "eng_Latn-mya_Mymr",
    "eng_Latn-nld_Latn",
    "eng_Latn-nob_Latn",
    "eng_Latn-npi_Deva",
    "eng_Latn-nso_Latn",
    "eng_Latn-pbt_Arab",
    "eng_Latn-pes_Arab",
    "eng_Latn-pol_Latn",
    "eng_Latn-por_Latn",
    "eng_Latn-ron_Latn",
    "eng_Latn-rus_Cyrl",
    "eng_Latn-sin_Sinh",
    "eng_Latn-slk_Latn",
    "eng_Latn-slv_Latn",
    "eng_Latn-som_Latn",
    "eng_Latn-spa_Latn",
    "eng_Latn-ssw_Latn",
    "eng_Latn-swe_Latn",
    "eng_Latn-swh_Latn",
    "eng_Latn-tam_Taml",
    "eng_Latn-tel_Telu",
    "eng_Latn-tgl_Latn",
    "eng_Latn-tha_Thai",
    "eng_Latn-tir_Ethi",
    "eng_Latn-tsn_Latn",
    "eng_Latn-tur_Latn",
    "eng_Latn-ukr_Cyrl",
    "eng_Latn-urd_Arab",
    "eng_Latn-vie_Latn",
    "eng_Latn-xho_Latn",
    "eng_Latn-yor_Latn",
    "eng_Latn-zho_Hans",
    "eng_Latn-zsm_Latn",
    "eng_Latn-zul_Latn",
    "est_Latn-eng_Latn",
    "ewe_Latn-fra_Latn",
    "fin_Latn-eng_Latn",
    "fon_Latn-fra_Latn",
    "fra_Latn-bam_Latn",
    "fra_Latn-eng_Latn",
    "fra_Latn-ewe_Latn",
    "fra_Latn-fon_Latn",
    "fra_Latn-mos_Latn",
    "fra_Latn-wol_Latn",
    "fuv_Latn-eng_Latn",
    "gaz_Latn-eng_Latn",
    "glg_Latn-eng_Latn",
    "guj_Gujr-eng_Latn",
    "hau_Latn-eng_Latn",
    "heb_Hebr-eng_Latn",
    "hin_Deva-eng_Latn",
    "hrv_Latn-eng_Latn",
    "hun_Latn-eng_Latn",
    "hye_Armn-eng_Latn",
    "ibo_Latn-eng_Latn",
    "ind_Latn-eng_Latn",
    "ita_Latn-eng_Latn",
    "jpn_Jpan-eng_Latn",
    "kat_Geor-eng_Latn",
    "kaz_Cyrl-eng_Latn",
    "khk_Cyrl-eng_Latn",
    "khm_Khmr-eng_Latn",
    "kmr_Latn-eng_Latn",
    "kor_Hang-eng_Latn",
    "lin_Latn-eng_Latn",
    "lit_Latn-eng_Latn",
    "lug_Latn-eng_Latn",
    "luo_Latn-eng_Latn",
    "lvs_Latn-eng_Latn",
    "mal_Mlym-eng_Latn",
    "mar_Deva-eng_Latn",
    "mkd_Cyrl-eng_Latn",
    "mos_Latn-fra_Latn",
    "mya_Mymr-eng_Latn",
    "nld_Latn-eng_Latn",
    "nob_Latn-eng_Latn",
    "npi_Deva-eng_Latn",
    "nso_Latn-eng_Latn",
    "pbt_Arab-eng_Latn",
    "pes_Arab-eng_Latn",
    "pol_Latn-eng_Latn",
    "por_Latn-eng_Latn",
    "ron_Latn-eng_Latn",
    "rus_Cyrl-eng_Latn",
    "sin_Sinh-eng_Latn",
    "slk_Latn-eng_Latn",
    "slv_Latn-eng_Latn",
    "som_Latn-eng_Latn",
    "spa_Latn-eng_Latn",
    "ssw_Latn-eng_Latn",
    "swe_Latn-eng_Latn",
    "swh_Latn-eng_Latn",
    "tam_Taml-eng_Latn",
    "tel_Telu-eng_Latn",
    "tgl_Latn-eng_Latn",
    "tha_Thai-eng_Latn",
    "tir_Ethi-eng_Latn",
    "tsn_Latn-eng_Latn",
    "tur_Latn-eng_Latn",
    "ukr_Cyrl-eng_Latn",
    "urd_Arab-eng_Latn",
    "vie_Latn-eng_Latn",
    "wol_Latn-fra_Latn",
    "xho_Latn-eng_Latn",
    "yor_Latn-eng_Latn",
    "zho_Hans-eng_Latn",
    "zsm_Latn-eng_Latn",
    "zul_Latn-eng_Latn",
]


def get_parser():
    parser = argparse.ArgumentParser(
        description="Tabulates BLEU & CHRF++ scores on the non-flores evaluation benchmarks"
    )
    # fmt: off
    parser.add_argument(
        '--output-dir',
        type=str,
        help='model dir'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint_best',
        help='checkpoint for which to tabulate the scores'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='valid',
        help='data split'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default="moe",
        help='output file'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='output file'
    )
    # fmt: on
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    model_type = args.model_type
    output_dir = args.output_dir
    results_dir = args.results_dir
    chk = args.checkpoint
    split = args.split

    if model_type == "moe":
        cap = "1.0"
    else:
        cap = "no_cap"

    bleus_map = {}
    chrf_map = {}
    sacrebleus_map = {}
    evaluated_lang_pairs = []

    for lang_pair in lang_pairs:
        # rtype = get_type(lang_pair)
        # print(f'Task {lang_pair} - Level {rtype}')
        src, tgt = lang_pair.split("-")
        pair_dir = f"{src}-{tgt}_{chk}_{split}"
        for datalabel in datalabels:
            out_dir = os.path.join(output_dir, f"gen_output_{datalabel}_{cap}")
            bleu_fpath = os.path.join(out_dir, pair_dir, "bleu.results")
            if os.path.exists(bleu_fpath):
                evaluated_lang_pairs.append(f"{lang_pair}-{datalabel}")
                command = f"cat {bleu_fpath} | cut -d' ' -f3 "
                bleu_str = subprocess.check_output(command, shell=True).decode()
                bleu = round(float(bleu_str), 2) if len(bleu_str) > 0 else -1
                bleus_map[f"{lang_pair}-{datalabel}"] = bleu

            chrf_fpath = os.path.join(out_dir, pair_dir, "chrf.results")
            if os.path.exists(chrf_fpath):
                command = f"grep score {chrf_fpath} | cut -f3 -d' ' | cut -f1 -d','"
                chrf_str = subprocess.check_output(command, shell=True).decode()
                chrf = round(float(chrf_str), 2) if len(chrf_str) > 0 else -1
                chrf_map[f"{lang_pair}-{datalabel}"] = chrf

            sacrebleu_fpath = os.path.join(out_dir, pair_dir, "sacrebleu.results")
            if os.path.exists(sacrebleu_fpath):
                command = f"cat {sacrebleu_fpath} | cut -d' ' -f3 "
                sacrebleu_str = subprocess.check_output(command, shell=True).decode()
                sacrebleu = (
                    round(float(sacrebleu_str), 2) if len(sacrebleu_str) > 0 else -1
                )
                sacrebleus_map[f"{lang_pair}-{datalabel}"] = sacrebleu

    # average_bleus = get_averages(bleus_map)
    # average_sacrebleus = get_averages(sacrebleus_map)
    # average_chrfs = get_averages(chrf_map)

    print("bleus_map keys:", list(bleus_map))
    for metric in ["bleu", "sacrebleu", "chrf"]:
        output_fpath = os.path.join(
            results_dir or output_dir, f"{metric}_{chk}_non_flores_{split}.tsv"
        )
        print(f"Writing results to {output_fpath}")
        with open(output_fpath, "w") as fout:
            if metric == "bleu":
                # average_vals = average_bleus
                metric_map = bleus_map
            elif metric == "sacrebleu":
                # average_vals = average_sacrebleus
                metric_map = sacrebleus_map
            else:
                # average_vals = average_chrfs
                metric_map = chrf_map

            # for subset, dict_values in average_vals.items():
            # for k, v in dict_values.items():
            # print(f"{subset}_{k}\t{v}", file=fout)
            for pair in evaluated_lang_pairs:
                if pair in metric_map:
                    print(f"{pair}\t{metric_map[pair]}", file=fout)


if __name__ == "__main__":
    main()
