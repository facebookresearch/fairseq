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
EXTRA_LANGS = [lang for lang in VP_LANGS if lang not in set(["en", "es", "fr"])]


# ============== HuBERT ==============
lang2fam = {
    "de": "germanic",
    "nl": "germanic",
    "sv": "germanic",
    "fi": "uralic",
    "et": "uralic",
    "hu": "uralic",
    "ca": "roman",
    "it": "roman",
    "pt": "roman",
    "ro": "roman",
    "cs": "slavic",
    "pl": "slavic",
    "hr": "slavic",
    "lt": "slavic",
    "lv": "slavic",
    "sk": "slavic",
    "sl": "slavic",
}
hubert_config_hub = {lang: (3, 11, 1000) for lang in VP_LANGS}
hubert_config_hub["it"] = (3, 11, 800)
en_es_fr_hubert = "mhubert_base_vp_en_es_fr_it3.pt"
hubert_model_hub = {
    "en": en_es_fr_hubert,
    "es": en_es_fr_hubert,
    "fr": en_es_fr_hubert,
}
for lang in EXTRA_LANGS:
    fam = lang2fam[lang]
    hubert_model_hub[lang] = f"mhubert_base_vp_{fam}_it3.pt"


# ============== kmeans ==============
en_es_fr_kmeans = "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"
kmeans_hub = {}
for lang in ["en", "es", "fr"]:
    kmeans_hub[lang] = en_es_fr_kmeans
for lang in EXTRA_LANGS:
    fam = lang2fam[lang]
    it, layer, km = hubert_config_hub[lang]
    kmeans_hub[lang] = f"mhubert_base_vp_{lang}_it{it}_L{layer}_km{km}.bin"
