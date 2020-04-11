import yaml
import sys
import os
from itertools import groupby

yaml_path = sys.argv[1]
with open(yaml_path) as f:
    wav_list = yaml.load(f, yaml.FullLoader)
    for wav_file_name, g in groupby(wav_list, lambda x: x['wav']):
        segment_list = list(g)
        wav_prefix = wav_file_name.split('.')[0]
        idx = wav_prefix.split("_")[1]
        for i, s in enumerate(segment_list):
            wav_name = f'{wav_prefix}_{i}.wav'
            wav_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(yaml_path),
                    "..",
                    "segmented_wav",
                    idx,
                    wav_name
                )
            )
            sys.stdout.write(f"{wav_path}\n")
