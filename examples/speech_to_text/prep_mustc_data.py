#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from tempfile import NamedTemporaryFile
import os
import os.path as op
import shutil
from typing import Tuple
from itertools import groupby

from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from torch import Tensor

from examples.speech_to_text.data_utils import (
    gen_vocab, create_zip, get_zip_manifest, save_df_to_tsv,
    extract_fbank_features, gen_config_yaml, filter_manifest_df
)

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ['id', 'audio', 'n_frames', 'tgt_text', 'speaker']
TASKS = ['asr', 'st']


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """
    SPLITS = ['train', 'dev', 'tst-COMMON', 'tst-HE']
    LANGUAGES = ['de', 'es', 'fr', 'it', 'nl', 'pt', 'ro', 'ru']

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = op.join(root, f'en-{lang}', 'data', split)
        wav_root, txt_root = op.join(_root, 'wav'), op.join(_root, 'txt')
        assert op.isdir(_root) and op.isdir(wav_root) and op.isdir(txt_root)
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print('Please install PyYAML to load YAML files for '
                  'the MuST-C dataset')
        with open(op.join(txt_root, f'{split}.yaml')) as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ['en', lang]:
            with open(op.join(txt_root, f'{split}.{_lang}')) as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x['wav']):
            wav_path = op.join(wav_root, wav_filename)
            sample_rate = torchaudio.info(wav_path)[0].rate
            seg_group = sorted(_seg_group, key=lambda x: x['offset'])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment['offset']) * sample_rate)
                n_frames = int(float(segment['duration']) * sample_rate)
                _id = f'{op.splitext(wav_filename)[0]}_{i}'
                self.data.append(
                    (wav_path, offset, n_frames, sample_rate, segment['en'],
                     segment[lang], segment['speaker_id'], _id)
                )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = \
            self.data[n]
        waveform, _ = torchaudio.load(wav_path, offset=offset,
                                      num_frames=n_frames)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    for lang in MUSTC.LANGUAGES:
        cur_root = op.join(args.data_root, f'en-{lang}')
        if not op.isdir(cur_root):
            print(f'{cur_root} does not exist. Skipped.')
            continue
        # Extract features
        feature_root = op.join(cur_root, 'fbank80')
        os.makedirs(feature_root, exist_ok=True)
        for split in MUSTC.SPLITS:
            print(f'Fetching split {split}...')
            dataset = MUSTC(args.data_root, lang, split)
            print('Extracting log mel filter bank features...')
            for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                extract_fbank_features(waveform, sample_rate,
                                       op.join(feature_root, f'{utt_id}.npy'))
        # Pack features into ZIP
        zip_filename = 'fbank80.zip'
        zip_path = op.join(cur_root, zip_filename)
        print('ZIPing features...')
        create_zip(feature_root, zip_path)
        print('Fetching ZIP manifest...')
        zip_manifest = get_zip_manifest(args.data_root,
                                        f'en-{lang}/{zip_filename}')
        # Generate TSV manifest
        print('Generating manifest...')
        train_text = {task: [] for task in TASKS}
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith('train')
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            text = {task: [] for task in TASKS}
            dataset = MUSTC(args.data_root, lang, split)
            for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
                manifest['id'].append(utt_id)
                manifest['audio'].append(zip_manifest[utt_id])
                duration_ms = int(wav.size(1) / sr * 1000)
                manifest['n_frames'].append(int(1 + (duration_ms - 25) / 10))
                text['asr'].append(src_utt)
                text['st'].append(tgt_utt)
                manifest['speaker'].append(speaker_id)
            if is_train_split:
                for task in TASKS:
                    train_text[task].extend(text[task])
            for task in TASKS:
                manifest['tgt_text'] = text[task]
                df = pd.DataFrame.from_dict(manifest)
                df = filter_manifest_df(df, is_train_split=is_train_split)
                save_df_to_tsv(df, op.join(cur_root, f'{split}_{task}.tsv'))
        # Generate vocab
        for task in TASKS:
            vocab_type, vocab_size = args.asr_vocab_type, args.asr_vocab_size
            if task == 'st':
                vocab_type, vocab_size = args.st_vocab_type, args.st_vocab_size
            vocab_size_str = '' if vocab_type == 'char' else str(vocab_size)
            spm_filename_prefix = f'spm_{vocab_type}{vocab_size_str}_{task}'
            with NamedTemporaryFile(mode='w') as f:
                for t in train_text[task]:
                    f.write(t + '\n')
                gen_vocab(f.name, op.join(cur_root, spm_filename_prefix),
                          vocab_type, vocab_size)
            # Generate config YAML
            gen_config_yaml(cur_root, spm_filename_prefix + '.model',
                            yaml_filename=f'config_{task}.yaml',
                            specaugment_policy='lb')
        # Clean up
        shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', '-d', required=True, type=str)
    parser.add_argument('--asr-vocab-type', default='unigram', required=True,
                        type=str, choices=['bpe', 'unigram', 'char']),
    parser.add_argument('--st-vocab-type', default='unigram', required=True,
                        type=str, choices=['bpe', 'unigram', 'char']),
    parser.add_argument('--asr-vocab-size', default=5000, type=int)
    parser.add_argument('--st-vocab-size', default=8000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == '__main__':
    main()
