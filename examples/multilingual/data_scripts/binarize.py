import shutil
import os, sys
from subprocess import check_call, check_output
import glob
import argparse
import shutil
import pathlib
import itertools

def call_output(cmd):
    print(f"Executing: {cmd}")
    ret = check_output(cmd, shell=True)
    print(ret)
    return ret

def call(cmd):
    print(cmd)
    check_call(cmd, shell=True)


WORKDIR_ROOT = os.environ.get('WORKDIR_ROOT', None)

if WORKDIR_ROOT is None or  not WORKDIR_ROOT.strip():
    print('please specify your working directory root in OS environment variable WORKDIR_ROOT. Exitting..."')
    sys.exit(-1)

SPM_PATH = os.environ.get('SPM_PATH', None)

if SPM_PATH is None or not SPM_PATH.strip():
    print("Please install sentence piecence from https://github.com/google/sentencepiece and set SPM_PATH pointing to the installed spm_encode.py. Exitting...")
    sys.exit(-1)


SPM_MODEL = f'{WORKDIR_ROOT}/sentence.bpe.model'
SPM_VOCAB = f'{WORKDIR_ROOT}/dict_250k.txt'

SPM_ENCODE = f'{SPM_PATH}'

if not os.path.exists(SPM_MODEL):
    call(f"wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/sentence.bpe.model -O {SPM_MODEL}")


if not os.path.exists(SPM_VOCAB):
    call(f"wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/dict_250k.txt -O {SPM_VOCAB}")



def get_data_size(raw):
    cmd = f'wc -l {raw}'
    ret = call_output(cmd)
    return int(ret.split()[0])

def encode_spm(model, direction, prefix='', splits=['train', 'test', 'valid'], pairs_per_shard=None):
    src, tgt = direction.split('-')

    for split in splits:
        src_raw, tgt_raw = f'{RAW_DIR}/{split}{prefix}.{direction}.{src}', f'{RAW_DIR}/{split}{prefix}.{direction}.{tgt}'
        if os.path.exists(src_raw) and os.path.exists(tgt_raw):
            cmd = f"""python {SPM_ENCODE} \
            --model {model}\
            --output_format=piece \
            --inputs {src_raw} {tgt_raw}  \
            --outputs {BPE_DIR}/{direction}{prefix}/{split}.bpe.{src} {BPE_DIR}/{direction}{prefix}/{split}.bpe.{tgt} """
            print(cmd)
            call(cmd)


def binarize_(
    bpe_dir,
    databin_dir,
    direction, spm_vocab=SPM_VOCAB, 
    splits=['train', 'test', 'valid'],
):
    src, tgt = direction.split('-')

    try:
        shutil.rmtree(f'{databin_dir}', ignore_errors=True)
        os.mkdir(f'{databin_dir}')
    except OSError as error:
        print(error)
    cmds = [
        "fairseq-preprocess",
        f"--source-lang {src} --target-lang {tgt}",
        f"--destdir {databin_dir}/",
        f"--workers 8",
    ]
    if isinstance(spm_vocab, tuple):
        src_vocab, tgt_vocab = spm_vocab
        cmds.extend(
            [
                f"--srcdict {src_vocab}",
                f"--tgtdict {tgt_vocab}",
            ]
        )
    else:
        cmds.extend(
            [
                f"--joined-dictionary",
                f"--srcdict {spm_vocab}",
            ]
        )
    input_options = []
    if 'train' in splits and glob.glob(f"{bpe_dir}/train.bpe*"):
        input_options.append(
            f"--trainpref {bpe_dir}/train.bpe",
        )        
    if 'valid' in splits and glob.glob(f"{bpe_dir}/valid.bpe*"):
        input_options.append(f"--validpref {bpe_dir}/valid.bpe")
    if 'test' in splits and glob.glob(f"{bpe_dir}/test.bpe*"):
        input_options.append(f"--testpref {bpe_dir}/test.bpe")   
    if len(input_options) > 0:    
        cmd = " ".join(cmds + input_options)
        print(cmd)
        call(cmd)


def binarize(
    databin_dir,
    direction, spm_vocab=SPM_VOCAB, prefix='',
    splits=['train', 'test', 'valid'],
    pairs_per_shard=None,
):
    def move_databin_files(from_folder, to_folder):
        for bin_file in glob.glob(f"{from_folder}/*.bin") \
            +  glob.glob(f"{from_folder}/*.idx") \
            +  glob.glob(f"{from_folder}/dict*"):
            try:
                shutil.move(bin_file, to_folder)
            except OSError as error:
                print(error)      
    bpe_databin_dir = f"{BPE_DIR}/{direction}{prefix}_databin"
    bpe_dir = f"{BPE_DIR}/{direction}{prefix}"
    if pairs_per_shard is None:
        binarize_(bpe_dir, bpe_databin_dir, direction, spm_vocab=spm_vocab, splits=splits)
        move_databin_files(bpe_databin_dir, databin_dir)
    else:
        # binarize valid and test which will not be sharded
        binarize_(
            bpe_dir, bpe_databin_dir, direction,
            spm_vocab=spm_vocab, splits=[s for s in splits if s != "train"])
        for shard_bpe_dir in glob.glob(f"{bpe_dir}/shard*"):
            path_strs = os.path.split(shard_bpe_dir)
            shard_str = path_strs[-1]
            shard_folder = f"{bpe_databin_dir}/{shard_str}"
            databin_shard_folder = f"{databin_dir}/{shard_str}"
            print(f'working from {shard_folder} to {databin_shard_folder}')
            os.makedirs(databin_shard_folder, exist_ok=True)
            binarize_(
                shard_bpe_dir, shard_folder, direction,
                spm_vocab=spm_vocab, splits=["train"])

            for test_data in glob.glob(f"{bpe_databin_dir}/valid.*") + glob.glob(f"{bpe_databin_dir}/test.*"):
                filename = os.path.split(test_data)[-1]
                try:
                    os.symlink(test_data, f"{databin_shard_folder}/{filename}")
                except OSError as error:
                    print(error)                
            move_databin_files(shard_folder, databin_shard_folder)


def load_langs(path):
    with open(path) as fr:
        langs = [l.strip() for l in fr]
    return langs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=f"{WORKDIR_ROOT}/ML50")
    parser.add_argument("--raw-folder", default='raw')
    parser.add_argument("--bpe-folder", default='bpe')    
    parser.add_argument("--databin-folder", default='databin')    

    args = parser.parse_args()

    DATA_PATH = args.data_root #'/private/home/yuqtang/public_data/ML50'   
    RAW_DIR = f'{DATA_PATH}/{args.raw_folder}'
    BPE_DIR = f'{DATA_PATH}/{args.bpe_folder}'
    DATABIN_DIR = f'{DATA_PATH}/{args.databin_folder}'
    os.makedirs(BPE_DIR, exist_ok=True)

    raw_files = itertools.chain(
        glob.glob(f'{RAW_DIR}/train*'),
        glob.glob(f'{RAW_DIR}/valid*'),
        glob.glob(f'{RAW_DIR}/test*'),
    )

    directions = [os.path.split(file_path)[-1].split('.')[1] for file_path in raw_files]

    for direction in directions:
        prefix = ""
        splits = ['train', 'valid', 'test']
        try:
            shutil.rmtree(f'{BPE_DIR}/{direction}{prefix}', ignore_errors=True)
            os.mkdir(f'{BPE_DIR}/{direction}{prefix}')
            os.makedirs(DATABIN_DIR, exist_ok=True)
        except OSError as error: 
            print(error)     
        spm_model, spm_vocab = SPM_MODEL, SPM_VOCAB
        encode_spm(spm_model, direction=direction, splits=splits)
        binarize(DATABIN_DIR, direction, spm_vocab=spm_vocab, splits=splits)
