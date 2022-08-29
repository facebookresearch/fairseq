# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import tqdm
from joblib import Parallel, delayed

from examples.nllb.evaluation.tokenizers import tokenize
from examples.nllb.modeling.evaluation.generate_multi import get_averages

logger = logging.getLogger(__name__)


def generate_results(direction, args):
    os.makedirs(args.output_dir, exist_ok=True)

    src, tgt = direction
    ref = get_reference_filename(args, src, tgt)
    hyp = get_hyp_filename(args, src, tgt)
    prep_cmd = ""

    if args.metric == "spbleu_flores101":
        sacrebleu_cmd = "-tok spm"
    # TODO(vedanuj) : change once sacrebleu has spbleu spm versioning
    elif args.metric == "spbleu_flores200":
        sacrebleu_cmd = "-tok spm"
    elif args.metric == "chrf":
        sacrebleu_cmd = "-m chrf"
    elif args.metric == "chrf++":
        sacrebleu_cmd = "-m chrf --chrf-word-order 2"
    elif args.metric == "tok_bleu":
        # For non-floes preprocess the irregular directions
        ref, hyp, prep_cmd, sacrebleu_cmd = tokenize(args, src, tgt, ref, hyp)
    elif args.metric == "bleu":
        # No special tokenization
        sacrebleu_cmd = "-tok 13a"
    else:
        raise ValueError(f"Unknown metric {args.metric}")

    cmd = (
        f"{prep_cmd}\n sacrebleu {sacrebleu_cmd} {ref} < {hyp} "
        + f" > {args.output_dir}/{src}-{tgt}.{args.metric}"
    )
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) as proc:
        if not args.quiet:
            logger.info(proc.stdout.read().decode("utf-8"))


def get_hyp_filename(args, src, tgt):
    corpus = args.corpus
    if "flores" in corpus and corpus != "floresv1":
        hyp = f"flores200-{src}-{tgt}-{args.split}.hyp"
    else:
        # non-flores references
        hyp = f"{args.corpus}-{src}-{tgt}-{args.split}.hyp"
    return f"{args.translate_dir}/{hyp}"


def get_reference_filename(args, src, tgt):
    corpus = args.corpus
    if "flores" in corpus and corpus != "floresv1":
        ref = f"flores200/{tgt}.devtest"
    else:
        # non-flores references
        if corpus == "autshumato" and tgt != "eng_Latn":  # multiway
            src = "eng_Latn"
        if tgt > src:
            ref = f"{corpus}/test.{src}-{tgt}.{tgt}"
        else:
            ref = f"{corpus}/test.{tgt}-{src}.{tgt}"
    return f"{args.reference_dir}/{ref}"


def get_directions(args):
    """
    Returns the list of directions for which we will look for generated
    hypotheses and evaluate the specified metric
    """
    # Table 30 first half
    if args.corpus == "flores101_87":
        langs = [
            "afr_Latn",
            "amh_Ethi",
            "arb_Arab",
            "ast_Latn",
            "azj_Latn",
            "bel_Cyrl",
            "bul_Cyrl",
            "ben_Beng",
            "bos_Latn",
            "cat_Latn",
            "ceb_Latn",
            "ces_Latn",
            "cym_Latn",
            "dan_Latn",
            "deu_Latn",
            "ell_Grek",
            "eng_Latn",
            "spa_Latn",
            "est_Latn",
            "prs_Arab",
            "fuv_Latn",
            "fin_Latn",
            "fra_Latn",
            "gle_Latn",
            "glg_Latn",
            "guj_Gujr",
            "hau_Latn",
            "heb_Hebr",
            "hin_Deva",
            "hrv_Latn",
            "hun_Latn",
            "hye_Armn",
            "ind_Latn",
            "ibo_Latn",
            "isl_Latn",
            "ita_Latn",
            "jpn_Jpan",
            "jav_Latn",
            "kat_Geor",
            "kaz_Cyrl",
            "khm_Khmr",
            "kan_Knda",
            "kor_Hang",
            "ltz_Latn",
            "lug_Latn",
            "lin_Latn",
            "lao_Laoo",
            "lit_Latn",
            "lvs_Latn",
            "mkd_Cyrl",
            "mal_Mlym",
            "khk_Cyrl",
            "mar_Deva",
            "zsm_Latn",
            "mya_Mymr",
            "npi_Deva",
            "nld_Latn",
            "nno_Latn",
            "nso_Latn",
            "oci_Latn",
            "ory_Orya",
            "pan_Guru",
            "pol_Latn",
            "pbt_Arab",
            "por_Latn",
            "ron_Latn",
            "rus_Cyrl",
            "sin_Sinh",
            "slk_Latn",
            "slv_Latn",
            "som_Latn",
            "srp_Cyrl",
            "swe_Latn",
            "swh_Latn",
            "tam_Taml",
            "tha_Thai",
            "tgl_Latn",
            "tur_Latn",
            "ukr_Cyrl",
            "urd_Arab",
            "uzn_Latn",
            "vie_Latn",
            "wol_Latn",
            "xho_Latn",
            "yor_Latn",
            "zho_Hans",
            "zul_Latn",
        ]
        directions = list(itertools.combinations(langs, r=2))
        rev_directions = [reversed(d) for d in directions]
        return directions + rev_directions

    # Table 30 second half - 10k directions
    elif args.corpus == "flores101":
        langs = [
            "afr_Latn",
            "amh_Ethi",
            "arb_Arab",
            "asm_Beng",
            "ast_Latn",
            "azj_Latn",
            "bel_Cyrl",
            "ben_Beng",
            "bos_Latn",
            "bul_Cyrl",
            "cat_Latn",
            "ceb_Latn",
            "ces_Latn",
            "ckb_Arab",
            "cym_Latn",
            "dan_Latn",
            "deu_Latn",
            "ell_Grek",
            "eng_Latn",
            "est_Latn",
            "fin_Latn",
            "fra_Latn",
            "fuv_Latn",
            "gaz_Latn",
            "gle_Latn",
            "glg_Latn",
            "guj_Gujr",
            "hau_Latn",
            "heb_Hebr",
            "hin_Deva",
            "hrv_Latn",
            "hun_Latn",
            "hye_Armn",
            "ibo_Latn",
            "ind_Latn",
            "isl_Latn",
            "ita_Latn",
            "jav_Latn",
            "jpn_Jpan",
            "kam_Latn",
            "kan_Knda",
            "kat_Geor",
            "kaz_Cyrl",
            "kea_Latn",
            "khk_Cyrl",
            "khm_Khmr",
            "kir_Cyrl",
            "kor_Hang",
            "lao_Laoo",
            "lin_Latn",
            "lit_Latn",
            "ltz_Latn",
            "lug_Latn",
            "luo_Latn",
            "lvs_Latn",
            "mal_Mlym",
            "mar_Deva",
            "mkd_Cyrl",
            "mlt_Latn",
            "mri_Latn",
            "mya_Mymr",
            "nld_Latn",
            "nob_Latn",
            "npi_Deva",
            "nso_Latn",
            "nya_Latn",
            "oci_Latn",
            "ory_Orya",
            "pan_Guru",
            "pbt_Arab",
            "pes_Arab",
            "pol_Latn",
            "por_Latn",
            "ron_Latn",
            "rus_Cyrl",
            "slk_Latn",
            "slv_Latn",
            "sna_Latn",
            "snd_Arab",
            "som_Latn",
            "spa_Latn",
            "srp_Cyrl",
            "swe_Latn",
            "swh_Latn",
            "tam_Taml",
            "tel_Telu",
            "tgk_Cyrl",
            "tgl_Latn",
            "tha_Thai",
            "tur_Latn",
            "ukr_Cyrl",
            "umb_Latn",
            "urd_Arab",
            "uzn_Latn",
            "vie_Latn",
            "wol_Latn",
            "xho_Latn",
            "yor_Latn",
            "zho_Hans",
            "zho_Hant",
            "zsm_Latn",
            "zul_Latn",
        ]
        directions = list(itertools.combinations(langs, r=2))
        rev_directions = [reversed(d) for d in directions]
        return directions + rev_directions
    # Table 31
    elif args.corpus == "flores101_african_en":
        langs = [
            "hau_Latn",
            "ibo_Latn",
            "lug_Latn",
            "luo_Latn",
            "swh_Latn",
            "wol_Latn",
            "xho_Latn",
            "yor_Latn",
            "zul_Latn",
        ]
        return [("eng_Latn", x) for x in langs] + [(x, "eng_Latn") for x in langs]

    # Table 53
    elif args.corpus == "flores101_african_ne":
        directions = [
            "ibo_Latn-swh_Latn",
            "ibo_Latn-xho_Latn",
            "ibo_Latn-yor_Latn",
            "ibo_Latn-fra_Latn",
            "swh_Latn-ibo_Latn",
            "swh_Latn-xho_Latn",
            "swh_Latn-yor_Latn",
            "swh_Latn-fra_Latn",
            "xho_Latn-ibo_Latn",
            "xho_Latn-swh_Latn",
            "xho_Latn-yor_Latn",
            "xho_Latn-fra_Latn",
            "yor_Latn-ibo_Latn",
            "yor_Latn-swh_Latn",
            "yor_Latn-xho_Latn",
            "yor_Latn-fra_Latn",
            "fra_Latn-ibo_Latn",
            "fra_Latn-swh_Latn",
            "fra_Latn-xho_Latn",
            "fra_Latn-yor_Latn",
        ]
        return [d.split("-") for d in directions]

    # Table 32
    elif args.corpus == "flores101_indic":
        langs = [
            "asm_Beng",
            "ben_Beng",
            "guj_Gujr",
            "hin_Deva",
            "kan_Knda",
            "mal_Mlym",
            "mar_Deva",
            "ory_Orya",
            "pan_Guru",
            "tam_Taml",
            "tel_Telu",
        ]
        return [("eng_Latn", x) for x in langs] + [(x, "eng_Latn") for x in langs]

    # Table 33
    elif args.corpus == "flores200":
        langs = [
            "ace_Arab",
            "ace_Latn",
            "acm_Arab",
            "acq_Arab",
            "aeb_Arab",
            "afr_Latn",
            "ajp_Arab",
            "aka_Latn",
            "amh_Ethi",
            "apc_Arab",
            "arb_Arab",
            "ars_Arab",
            "ary_Arab",
            "arz_Arab",
            "asm_Beng",
            "ast_Latn",
            "awa_Deva",
            "ayr_Latn",
            "azb_Arab",
            "azj_Latn",
            "bak_Cyrl",
            "bam_Latn",
            "ban_Latn",
            "bel_Cyrl",
            "bem_Latn",
            "ben_Beng",
            "bho_Deva",
            "bjn_Arab",
            "bjn_Latn",
            "bod_Tibt",
            "bos_Latn",
            "bug_Latn",
            "bul_Cyrl",
            "cat_Latn",
            "ceb_Latn",
            "ces_Latn",
            "cjk_Latn",
            "ckb_Arab",
            "crh_Latn",
            "cym_Latn",
            "dan_Latn",
            "deu_Latn",
            "dik_Latn",
            "dyu_Latn",
            "dzo_Tibt",
            "ell_Grek",
            "eng_Latn",
            "epo_Latn",
            "est_Latn",
            "eus_Latn",
            "ewe_Latn",
            "fao_Latn",
            "pes_Arab",
            "fij_Latn",
            "fin_Latn",
            "fon_Latn",
            "fra_Latn",
            "fur_Latn",
            "fuv_Latn",
            "gla_Latn",
            "gle_Latn",
            "glg_Latn",
            "grn_Latn",
            "guj_Gujr",
            "hat_Latn",
            "hau_Latn",
            "heb_Hebr",
            "hin_Deva",
            "hne_Deva",
            "hrv_Latn",
            "hun_Latn",
            "hye_Armn",
            "ibo_Latn",
            "ilo_Latn",
            "ind_Latn",
            "isl_Latn",
            "ita_Latn",
            "jav_Latn",
            "jpn_Jpan",
            "kab_Latn",
            "kac_Latn",
            "kam_Latn",
            "kan_Knda",
            "kas_Arab",
            "kas_Deva",
            "kat_Geor",
            "knc_Arab",
            "knc_Latn",
            "kaz_Cyrl",
            "kbp_Latn",
            "kea_Latn",
            "khm_Khmr",
            "kik_Latn",
            "kin_Latn",
            "kir_Cyrl",
            "kmb_Latn",
            "kon_Latn",
            "kor_Hang",
            "kmr_Latn",
            "lao_Laoo",
            "lvs_Latn",
            "lij_Latn",
            "lim_Latn",
            "lin_Latn",
            "lit_Latn",
            "lmo_Latn",
            "ltg_Latn",
            "ltz_Latn",
            "lua_Latn",
            "lug_Latn",
            "luo_Latn",
            "lus_Latn",
            "mag_Deva",
            "mai_Deva",
            "mal_Mlym",
            "mar_Deva",
            "min_Latn",
            "mkd_Cyrl",
            "plt_Latn",
            "mlt_Latn",
            "mni_Beng",
            "khk_Cyrl",
            "mos_Latn",
            "mri_Latn",
            "zsm_Latn",
            "mya_Mymr",
            "nld_Latn",
            "nno_Latn",
            "nob_Latn",
            "npi_Deva",
            "nso_Latn",
            "nus_Latn",
            "nya_Latn",
            "oci_Latn",
            "gaz_Latn",
            "ory_Orya",
            "pag_Latn",
            "pan_Guru",
            "pap_Latn",
            "pol_Latn",
            "por_Latn",
            "prs_Arab",
            "pbt_Arab",
            "quy_Latn",
            "ron_Latn",
            "run_Latn",
            "rus_Cyrl",
            "sag_Latn",
            "san_Deva",
            "sat_Olck",
            "scn_Latn",
            "shn_Mymr",
            "sin_Sinh",
            "slk_Latn",
            "slv_Latn",
            "smo_Latn",
            "sna_Latn",
            "snd_Arab",
            "som_Latn",
            "sot_Latn",
            "spa_Latn",
            "als_Latn",
            "srd_Latn",
            "srp_Cyrl",
            "ssw_Latn",
            "sun_Latn",
            "swe_Latn",
            "swh_Latn",
            "szl_Latn",
            "tam_Taml",
            "tat_Cyrl",
            "tel_Telu",
            "tgk_Cyrl",
            "tgl_Latn",
            "tha_Thai",
            "tir_Ethi",
            "taq_Latn",
            "taq_Tfng",
            "tpi_Latn",
            "tsn_Latn",
            "tso_Latn",
            "tuk_Latn",
            "tum_Latn",
            "tur_Latn",
            "twi_Latn",
            "tzm_Tfng",
            "uig_Arab",
            "ukr_Cyrl",
            "umb_Latn",
            "urd_Arab",
            "uzn_Latn",
            "vec_Latn",
            "vie_Latn",
            "war_Latn",
            "wol_Latn",
            "xho_Latn",
            "ydd_Hebr",
            "yor_Latn",
            "yue_Hant",
            "zho_Hans",
            "zho_Hant",
            "zul_Latn",
        ]
        directions = list(itertools.combinations(langs, r=2))
        rev_directions = [reversed(d) for d in directions]
        return directions + rev_directions

    # Table 34
    elif args.corpus == "flores200_102":
        pass  # TODO
    # Table 54
    elif args.corpus == "flores200_206":
        langs = [
            "afr_Latn",
            "als_Latn",
            "amh_Ethi",
            "arb_Arab",
            "azj_Latn",
            "bel_Cyrl",
            "ben_Beng",
            "bos_Latn",
            "bul_Cyrl",
            "cat_Latn",
            "ceb_Latn",
            "ces_Latn",
            "cym_Latn",
            "dan_Latn",
            "deu_Latn",
            "ell_Grek",
            "epo_Latn",
            "est_Latn",
            "eus_Latn",
            "fin_Latn",
            "fra_Latn",
            "gla_Latn",
            "gle_Latn",
            "glg_Latn",
            "guj_Gujr",
            "hat_Latn",
            "hau_Latn",
            "heb_Hebr",
            "hin_Deva",
            "hrv_Latn",
            "hun_Latn",
            "hye_Armn",
            "ibo_Latn",
            "ind_Latn",
            "isl_Latn",
            "ita_Latn",
            "jav_Latn",
            "jpn_Jpan",
            "kan_Knda",
            "kat_Geor",
            "kaz_Cyrl",
            "khk_Cyrl",
            "khm_Khmr",
            "kin_Latn",
            "kir_Cyrl",
            "kmr_Latn",
            "kor_Hang",
            "lao_Laoo",
            "lit_Latn",
            "ltz_Latn",
            "lvs_Latn",
            "mal_Mlym",
            "mar_Deva",
            "mkd_Cyrl",
            "mlt_Latn",
            "mri_Latn",
            "mya_Mymr",
            "nld_Latn",
            "nob_Latn",
            "npi_Deva",
            "nya_Latn",
            "ory_Orya",
            "pan_Guru",
            "pbt_Arab",
            "pes_Arab",
            "plt_Latn",
            "pol_Latn",
            "por_Latn",
            "ron_Latn",
            "rus_Cyrl",
            "sin_Sinh",
            "slk_Latn",
            "slv_Latn",
            "smo_Latn",
            "sna_Latn",
            "snd_Arab",
            "som_Latn",
            "sot_Latn",
            "spa_Latn",
            "srp_Cyrl",
            "sun_Latn",
            "swe_Latn",
            "swh_Latn",
            "tam_Taml",
            "tat_Cyrl",
            "tel_Telu",
            "tgk_Cyrl",
            "tgl_Latn",
            "tha_Thai",
            "tuk_Latn",
            "tur_Latn",
            "uig_Arab",
            "ukr_Cyrl",
            "urd_Arab",
            "uzn_Latn",
            "vie_Latn",
            "xho_Latn",
            "ydd_Hebr",
            "yor_Latn",
            "zho_Hans",
            "zho_Hant",
            "zsm_Latn",
            "zul_Latn",
        ]
        return [("eng_Latn", x) for x in langs] + [(x, "eng_Latn") for x in langs]

    # Table 38
    elif args.corpus == "mafand":
        langs_eng = [
            "hau_Latn",
            "ibo_Latn",
            "lug_Latn",
            "luo_Latn",
            "swh_Latn",
            "tsn_Latn",
            "yor_Latn",
            "zul_Latn",
        ]
        langs_fra = ["bam_Latn", "ewe_Latn", "fon_Latn", "mos_Latn", "wol_Latn"]
        return (
            [("eng_Latn", x) for x in langs_eng]
            + [(x, "eng_Latn") for x in langs_eng]
            + [("fra_Latn", x) for x in langs_fra]
            + [(x, "fra_Latn") for x in langs_fra]
        )

    # Table 58 lhs
    elif args.corpus == "madar":
        langs = [
            "ary_Arab",
            "acm_Arab",
            "apc_Arab",
            "ars_Arab",
            "acq_Arab",
            "ajp_Arab",
            "aeb_Arab",
            "arz_Arab",
        ]
        return [("arb_Arab", x) for x in langs] + [(x, "arb_Arab") for x in langs]

    # Table 57
    elif args.corpus == "autshumato":
        langs = [
            "eng_Latn",
            "afr_Latn",
            "nso_Latn",
            "sot_Latn",
            "ssw_Latn",
            "tso_Latn",
            "tsn_Latn",
            "xho_Latn",
            "zul_Latn",
        ]
        directions = list(itertools.combinations(langs, r=2))
        rev_directions = [reversed(d) for d in directions]
        return directions + rev_directions

    # English-centric
    # Table 35.a
    elif args.corpus == "floresv1":
        langs = ["khm_Khmr", "npi_Deva", "pbt_Arab", "sin_Sinh"]
    # Table 35.b
    elif args.corpus == "wat":
        langs = ["hin_Deva", "khm_Khmr", "mya_Mymr"]
    # Table 36.a
    elif args.corpus == "wmt":
        langs = [
            "ces_Latn",
            "deu_Latn",
            "est_Latn",
            "fin_Latn",
            "fra_Latn",
            "guj_Gujr",
            "hin_Deva",
            "kaz_Cyrl",
            "lit_Latn",
            "lvs_Latn",
            "ron_Latn",
            "rus_Cyrl",
            "spa_Latn",
            "tur_Latn",
            "zho_Hans",
        ]

    # Table 36.b
    elif args.corpus == "iwslt":
        langs = [
            "arb_Arab",
            "deu_Latn",
            "fra_Latn",
            "ita_Latn",
            "jpn_Jpan",
            "kor_Hang",
            "nld_Latn",
            "pes_Arab",
            "pol_Latn",
            "ron_Latn",
            "rus_Cyrl",
            "vie_Latn",
        ]

    # Table 37 & 58 rhs
    elif args.corpus == "tico":
        langs = [
            "arb_Arab",
            "fra_Latn",
            "gaz_Latn",
            "hin_Deva",
            "ind_Latn",
            "lin_Latn",
            "lug_Latn",
            "mar_Deva",
            "pes_Arab",
            "por_Latn",
            "rus_Cyrl",
            "spa_Latn",
            "swh_Latn",
            "urd_Arab",
            "zho_Hans",
            "zsm_Latn",
            "zul_Latn",  # main
            "amh_Ethi",
            "ben_Beng",
            "ckb_Arab",
            "hau_Latn",
            "kmr_Latn",
            "mya_Mymr",
            "npi_Deva",
            "pbt_Arab",
            "som_Latn",
            "tgl_Latn",
            "tir_Ethi",
        ]

    else:
        raise ValueError("unknown group {arg.corpus}")

    return [("eng_Latn", x) for x in langs] + [(x, "eng_Latn") for x in langs]


def print_averages(score_dict):
    average_bleus = pd.DataFrame(get_averages(score_dict))
    print(average_bleus)


def print_pivot_centric(score_dict, pivot_xx_directions):
    scores = pd.DataFrame(
        {
            "langs": [tgt for src, tgt in pivot_xx_directions],
            f"{args.pivot}-xx": [
                score_dict[f"{src}-{tgt}"] for src, tgt in pivot_xx_directions
            ],
            f"xx-{args.pivot}": [
                score_dict[f"{tgt}-{src}"] for src, tgt in pivot_xx_directions
            ],
        }
    )
    print(scores.to_string(index=False))


def print_multiway(score_dict, langs):
    directions = list(score_dict)
    df = pd.DataFrame(
        {"direction": directions, "score": [score_dict[d] for d in directions]}
    )
    df["source"] = df.apply(lambda x: x.direction.split("-")[0], axis=1)
    df["target"] = df.apply(lambda x: x.direction.split("-")[1], axis=1)
    df = df.pivot(index="source", columns="target", values="score")
    print(df[langs].reindex(langs))


def print_list(score_dict, directions):
    print(
        pd.DataFrame(
            {"direction": directions, "score": [score_dict[d] for d in directions]}
        )
    )


def aggregate_metrics(args):
    directions = get_directions(args)
    supervised_directions = Path(
        "examples/nllb/modeling/scripts/flores200/lang_pairs.txt"
    ).read_text()
    print(supervised_directions)
    supervised_directions = supervised_directions.split(",")
    score_dict = {}
    pivot_xx_directions = []
    for direction in tqdm.tqdm(directions):
        src, tgt = direction
        if args.zero_shot and args.supervised:
            raise ValueError("Cannot be both zero-shot and supervised.")
        if args.zero_shot and f"{src}-{tgt}" in supervised_directions:
            continue
        if args.supervised and f"{src}-{tgt}" not in supervised_directions:
            continue
        score_file = f"{args.output_dir}/{src}-{tgt}.{args.metric}"
        command = f"grep 'score' {score_file} | head -1 | cut -f3 -d' ' | cut -f1 -d','"
        value_str = subprocess.check_output(command, shell=True).decode()
        value = round(float(value_str), 2) if len(value_str) > 0 else -1
        score_dict[f"{src}-{tgt}"] = value
        if src == args.pivot:
            pivot_xx_directions.append((src, tgt))

    print(f"Metric :: {args.metric}")
    if args.average:
        print_averages(score_dict)
    else:
        if args.corpus == "autshumato":
            ordered_langs = [
                "eng_Latn",
                "afr_Latn",
                "nso_Latn",
                "sot_Latn",
                "ssw_Latn",
                "tso_Latn",
                "tsn_Latn",
                "xho_Latn",
                "zul_Latn",
            ]
            print_multiway(score_dict, ordered_langs)
        elif args.corpus == "flores101_african_ne":
            directions = [
                "ibo_Latn-swh_Latn",
                "ibo_Latn-xho_Latn",
                "ibo_Latn-yor_Latn",
                "ibo_Latn-fra_Latn",
                "swh_Latn-ibo_Latn",
                "swh_Latn-xho_Latn",
                "swh_Latn-yor_Latn",
                "swh_Latn-fra_Latn",
                "xho_Latn-ibo_Latn",
                "xho_Latn-swh_Latn",
                "xho_Latn-yor_Latn",
                "xho_Latn-fra_Latn",
                "yor_Latn-ibo_Latn",
                "yor_Latn-swh_Latn",
                "yor_Latn-xho_Latn",
                "yor_Latn-fra_Latn",
                "fra_Latn-ibo_Latn",
                "fra_Latn-swh_Latn",
                "fra_Latn-xho_Latn",
                "fra_Latn-yor_Latn",
            ]
            print_list(score_dict, directions)

        else:
            print_pivot_centric(score_dict, pivot_xx_directions)


def main(args):
    directions = get_directions(args)
    _ = Parallel(n_jobs=32)(
        delayed(generate_results)(direction, args)
        for direction in tqdm.tqdm(directions)
    )

    aggregate_metrics(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--translate-dir", default=None)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--pivot", default="eng_Latn")

    parser.add_argument("--metric", default="chrf")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quiet", default=True)
    parser.add_argument("--average", action="store_true")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--supervised", action="store_true")

    args = parser.parse_args()

    if args.corpus == "madar":
        args.pivot = "arb_Arab"
    args.split = "devtest" if "flores" in args.corpus else "test"
    if args.corpus == "floresv1":
        args.split = "test"
    main(args)
