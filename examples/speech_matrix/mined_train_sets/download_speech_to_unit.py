import os
import argparse
from examples.speech_matrix.data_helper.data_cfg import (
    DOWNLOAD_HUB,
    manifest_key,
)


def download_s2u_manifests(save_root):
    # download
    manifest_dl = f"{DOWNLOAD_HUB}/{manifest_key}.tar.gz"
    os.system(f"wget {manifest_dl} -P {save_root}")
    # unzip
    os.system(f"tar -zxvf {save_root}/{manifest_key}.tar.gz -C {save_root}/")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root", type=str, required=True)
    args = parser.parse_args()

    # download speech_to_unit manifests
    download_s2u_manifests(args.save_root)
