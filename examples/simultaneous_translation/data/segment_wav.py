import os
import json
import yaml
import soundfile as sf
from multiprocessing import cpu_count
from itertools import groupby
from tqdm import tqdm
import sentencepiece as sp
import argparse

SPLITS = ['train', 'dev', 'tst-COMMON', 'tst-HE']

def segment_wav_files(root, split):
    in_root = os.path.join(root, split, 'wav')
    out_root = os.path.join(root, split, 'segmented_wav')
    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    yaml_path = os.path.join(root, split, 'txt', f'{split}.yaml')
    with open(yaml_path) as f:
        wav_list = yaml.load(f)

    for wav_file_name, g in tqdm(groupby(wav_list, lambda x: x['wav'])):
        segment_list = list(g)

        wav_file_path = os.path.join(in_root, wav_file_name)
        ted_id = os.path.splitext(wav_file_name)[0].split('_')[1]
        out_dir_path = os.path.join(out_root, ted_id)
        if not os.path.isdir(out_dir_path):
            os.makedirs(out_dir_path)

        with sf.SoundFile(wav_file_path) as f:
            sr = f.samplerate
            for i, s in tqdm(enumerate(segment_list), total=len(segment_list)):
                offset, duration = round(s['offset'], 3), round(s['duration'], 3)
                f.seek(int(offset * sr))
                frames_to_read = int(duration * sr)
                out_file_name = os.path.splitext(wav_file_name)
                out_file_name = f'{out_file_name[0]}_{i}{out_file_name[1]}'
                out_file_path = os.path.join(out_dir_path, out_file_name)
                with sf.SoundFile(
                        out_file_path, 'w', samplerate=f.samplerate,
                        channels=f.channels, subtype=f.subtype, endian=f.endian,
                        format=f.format
                ) as f_out:
                    f_out.write(f.read(frames_to_read))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    args = parser.parse_args()
    for split in SPLITS:
        segment_wav_files(args.data_path, split)

