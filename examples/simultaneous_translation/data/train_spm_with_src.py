from pathlib import Path
import os
import json
from typing import List, Optional
import re
from subprocess import Popen, PIPE
from multiprocessing import cpu_count

import yaml
import sentencepiece as sp
from sacrebleu import tokenize_13a
from tqdm import tqdm
import argparse

UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
PAD = '<pad>'

FRAME_LEN = 25
FRAME_SHIFT = 10
MIN_FRAMES = 20
MAX_FRAMES = 3000
MIN_TOKENS = 1
MAX_TOKENS = 256
MAX_TEXT = 2

SPLITS = ['train', 'dev', 'tst-COMMON', 'tst-HE']
WHITESPACE_NORMALIZER = re.compile(r'\s+')
SPACE = chr(32)

def gen_dict_txt(in_path: str, out_dir_path: str, lang:str):
    out_path = Path(out_dir_path) / f'dict.{lang}.txt'
    with open(in_path) as f_in, open(out_path, 'w') as f_out:
        for l_in in f_in:
            s, c = l_in.strip().split()
            if s in {UNK, BOS, EOS, PAD}:
                continue
            f_out.write(f'{s} 1\n')
    return out_path

def train_spm_vocab(
    input_path: str, 
    out_root: str, 
    vocab_size: int,
    model_type: str, 
    lang: str
):
    if not Path(out_root).is_dir():
        os.makedirs(out_root)

    model_prefix = Path(out_root) / f'spm_{model_type}_{vocab_size}_{lang}'

    arguments = [
        f'--input={input_path}',
        f'--model_prefix={model_prefix}',
        f'--model_type={model_type}',
        f'--vocab_size={vocab_size}',
        '--character_coverage=1.0',
        f'--num_threads={cpu_count()}',
        '--unk_id=3',
        '--bos_id=0',
        '--eos_id=2',
        '--pad_id=1'
    ]
    sp.SentencePieceTrainer.Train(' '.join(arguments))
    return model_prefix

def build_dict(
    data_path: str, 
    out_path: str, 
    vocab_size: int, 
    model_type: str,
    lang: str
):
    model_prefix = train_spm_vocab(
        input_path=os.path.join(data_path, 'train', 'txt', f'train.{lang}'),
        out_root=os.path.join(out_path, f'vocab'),
        vocab_size=vocab_size,
        model_type=model_type,
        lang=lang,
    )

    try:
        os.unlink(os.path.join(out_path, f'spm.{lang}.model'))
    except:
        pass

    os.symlink(
        os.path.join(
            'vocab',
            os.path.basename(model_prefix.__str__())+ '.model'
        ),
        os.path.join(out_path, f'spm.{lang}.model')
    )

    gen_dict_txt(
        in_path=model_prefix.__str__()+ '.vocab',
        out_dir_path=os.path.join(out_path),
        lang=lang
    )

    return os.path.join(out_path, f'spm.{lang}.model') 

def gen_metadata_json(
    data_path: str,
    out_path: str,
    spm_model_paths: list, 
    langs:list,
    split:str
):
    spm_model={}
    for lang, model_path in zip(langs, spm_model_paths):
        spm_model[lang] = sp.SentencePieceProcessor()
        spm_model[lang].Load(model_path)
            
    yaml_path = os.path.join(data_path, split, 'txt', f'{split}.yaml')
    with open(yaml_path) as f:
        wav_list = yaml.load(f)

    src_path = os.path.join(data_path, split, 'txt', f'{split}.{langs[0]}')
    tgt_path = os.path.join(data_path, split, 'txt', f'{split}.{langs[1]}')

    prev_wav_name, count = None, 0
    output = {'utts': {}}
    n_filtered = 0
    with open(src_path) as f_src, open(tgt_path) as f_tgt:
        for s, t, w in tqdm(zip(f_src, f_tgt, wav_list), total=len(wav_list)):
            ss = s.strip()
            tt = t.strip()
            if prev_wav_name is None or w['wav'] != prev_wav_name:
                count = 0
                prev_wav_name = w['wav']
            ted_id = os.path.splitext(w['wav'])[0].split('_')[1]
            cur_id = '{}-{}-{}'.format(ted_id, ted_id, count)
            filename = f'ted_{ted_id}_{count}.wav'
            count += 1

            duration = round(w['duration'], 3)
            frame_len_ms = int(duration * 1000)
            n_frames = int(1 + (frame_len_ms - FRAME_LEN) / FRAME_SHIFT)

            tgt_tokens = spm_model[langs[1]].EncodeAsPieces(tt)
            tgt_token_ids = spm_model[langs[1]].EncodeAsIds(tt)
            src_tokens = spm_model[langs[0]].EncodeAsPieces(ss)
            src_token_ids = spm_model[langs[0]].EncodeAsIds(ss)

            assert len(tgt_tokens) == len(tgt_token_ids)
            assert len(src_tokens) == len(src_token_ids)
            if n_frames < MIN_FRAMES or len(tgt_tokens) < MIN_TOKENS:
                n_filtered += 1
                continue
            if split == 'train' and (n_frames > MAX_FRAMES or len(tgt_tokens) > MAX_TOKENS):
                n_filtered += 1
                continue

            output['utts'][cur_id] = {
                'input': {
                    'length_ms': frame_len_ms,
                    'path': os.path.join(
                        data_path, split, 'segmented_wav', ted_id, filename
                    ),
                },
                'output': {
                    'text': tt,
                    'token': ' '.join(tgt_tokens),
                    'tokenid': ', '.join(str(i) for i in tgt_token_ids),
                    'source':{
                        'text': ss,
                        'token': ' '.join(src_tokens),
                        'tokenid': ', '.join(str(i) for i in src_token_ids),
                    }
                }
            }
    print(f'{n_filtered}/{len(wav_list)} filtered out')

    with open(os.path.join(out_path, f'{split}.json'), 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--source-vocab-size", type=int, required=True)
    parser.add_argument("--target-vocab-size", type=int, required=True)
    parser.add_argument("--target-model-type", type=str, required=True)
    parser.add_argument("--source-model-type", type=str, required=True)
    parser.add_argument("--max-frame", type=int, default=3000)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--source-lang", type=str, required=True)
    parser.add_argument("--target-lang", type=str, required=True)
    args = parser.parse_args()

    output_name = f"{args.source_lang}-{args.source_model_type}-{args.source_vocab_size}-{args.target_lang}-{args.target_model_type}-{args.target_vocab_size}-{args.max_frame}"

    if not os.path.exists(os.path.join(args.out_path, output_name)):
        os.mkdir(os.path.join(args.out_path, output_name))
    
    out_path = os.path.join(args.out_path, output_name)

    source_spm = build_dict(
        data_path=args.data_path,
        out_path=out_path,
        vocab_size=args.source_vocab_size,
        model_type=args.source_model_type,
        lang=args.source_lang
    )

    target_spm = build_dict(
        data_path=args.data_path,
        out_path=out_path,
        vocab_size=args.target_vocab_size,
        model_type=args.target_model_type,
        lang=args.target_lang
    )

    for split in SPLITS:
        gen_metadata_json(
            data_path=os.path.abspath(os.path.join(args.data_path)),
            out_path=out_path,
            spm_model_paths=[source_spm, target_spm], 
            langs=[args.source_lang, args.target_lang],
            split=split
        )

    try:
        os.unlink(os.path.join(args.out_path, output_name, 'valid.json'))
    except:
        pass

    os.symlink(
        'dev.json',
        os.path.join(args.out_path, output_name, 'valid.json')
    )

    try:
        os.unlink(os.path.join(args.out_path, output_name, 'test.json'))
    except:
        pass

    os.symlink(
        'tst-COMMON.json',
        os.path.join(args.out_path, output_name, 'test.json')
    )


