# EuroParl-ST
EP_LANGS = ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ro"]


# VoxPopuli
VP_FAMS = [
    "germanic",
    "uralic",
]
VP_LANGS = [
    "cs",
    "de",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hr",
    "hu",
    "it",
    "lt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
]
VP_LANG_PAIRS = []
for src_lang in VP_LANGS:
    for tgt_lang in VP_LANGS:
        if src_lang == tgt_lang:
            continue
        VP_LANG_PAIRS.append(f"{src_lang}-{tgt_lang}")

high_res_pairs = set(
    [
        "en-es",
        "en-fr",
        "de-en",
        "en-it",
        "en-pl",
        "en-pt",
        "es-fr",
        "es-it",
        "es-pt",
        "fr-it",
        "fr-pt",
        "it-pt",
        "en-nl",
    ]
)
mid_res_pairs = set(
    [
        "en-ro",
        "cs-de",
        "de-es",
        "de-pl",
        "en-sk",
        "cs-fr",
        "es-nl",
        "en-fi",
        "cs-es",
        "en-hu",
        "cs-en",
        "hu-it",
        "nl-pt",
        "pl-pt",
        "nl-pl",
        "de-fr",
        "de-it",
        "fr-nl",
        "it-nl",
        "es-pl",
        "cs-pl",
        "de-nl",
        "es-ro",
        "it-pl",
        "fr-ro",
        "de-pt",
        "pl-sk",
        "cs-it",
        "it-ro",
        "cs-nl",
        "cs-sk",
        "fr-pl",
    ]
)


# FLEURS
FLEURS_LANGS = [
    "cs_cz",
    "de_de",
    "en_us",
    "es_419",
    "et_ee",
    "fi_fi",
    "fr_fr",
    "hr_hr",
    "hu_hu",
    "it_it",
    "lt_lt",
    "nl_nl",
    "pl_pl",
    "pt_br",
    "ro_ro",
    "sk_sk",
    "sl_si",
]
FLORES_LANG_MAP = {
    "cs": "ces",
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "et": "est",
    "fi": "fin",
    "fr": "fra",
    "hr": "hrv",
    "hu": "hun",
    "it": "ita",
    "lt": "lit",
    "nl": "nld",
    "pl": "pol",
    "pt": "por",
    "ro": "ron",
    "sk": "slk",
    "sl": "slv",
}

DOWNLOAD_HUB = "https://dl.fbaipublicfiles.com/speech_matrix"

manifest_prefix = "train_mined"

# sub directories under save_root/
audio_key = "audios"
aligned_speech_key = "aligned_speech"
manifest_key = "s2u_manifests"
hubert_key = "hubert"
vocoder_key = "vocoder"
s2s_key = "s2s_models"
