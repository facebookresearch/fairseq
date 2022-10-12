import os
from examples.hubert.simple_kmeans import (
    dump_hubert_feature,
    dump_km_label,
)
from examples.speech_matrix.data_helper.model_cfg import (
    hubert_config_hub,
    hubert_model_hub,
    kmeans_hub,
)


def extract_lang_units(
    aud_manifest,
    lang,
    km_save_dir,
    hubert_model_dir,
):
    manifest_dir = os.path.dirname(aud_manifest)
    split = os.path.basename(aud_manifest)
    if split.endswith(".tsv"):
        split = split[:-4]
    feat_dir = os.path.join(km_save_dir, "tmp_features")
    os.makedirs(feat_dir, exist_ok=True)

    # hubert feature
    it, layer, km = hubert_config_hub[lang]
    ckpt_path = os.path.join(hubert_model_dir, hubert_model_hub[lang])
    dump_hubert_feature.main(
        manifest_dir, split, ckpt_path, layer, 1, 0, feat_dir, max_chunk=1600000
    )

    # kmeans label
    km_path = os.path.join(hubert_model_dir, kmeans_hub[lang])
    dump_km_label.dump_label(feat_dir, split, km_path, 1, 0, km_save_dir)

    cmd = f"mv {km_save_dir}/{split}_0_1.km {km_save_dir}/{split}.km"
    os.system(cmd)

    os.system(f"rm {feat_dir}/{split}*")
    print(f"done unit extraction: {split}")
