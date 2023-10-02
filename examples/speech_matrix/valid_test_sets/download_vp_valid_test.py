import os
import sys
import argparse
from examples.speech_matrix.data_helper.data_cfg import (
    DOWNLOAD_HUB,
    audio_key
)

def download_data(save_root):
    save_dir = os.path.join(save_root, audio_key)
    aud_dl = f"{DOWNLOAD_HUB}/{audio_key}/valid_test_vp_aud.zip"
    os.system(f"wget {aud_dl} -P {save_dir}")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root", type=str, required=True)
    args = parser.parse_args()

    download_data(args.save_root)



