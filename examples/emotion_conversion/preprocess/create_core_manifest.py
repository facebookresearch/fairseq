from pathlib import Path
import os
import sys
import subprocess
import argparse
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler('debug.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def verify_dict_size(km, dict):
    logger.info(f"verifying: {km}")
    dict_size = len(open(dict, "r").readlines())
    km_vocab = set(open(km, "r").read().replace("\n", " ").split(" "))
    if "" in km_vocab: km_vocab.remove("")
    km_vocab_size = len(km_vocab)
    return dict_size == km_vocab_size


def verify_files_exist(l):
    for f in l:
        if not f.exists():
            logging.error(f"{f} doesn't exist!")
            return False
    return True


def run_cmd(cmd, print_output=True):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
        if print_output:
            logger.info(f"command output:\n{out}")
        return out
    except subprocess.CalledProcessError as grepexc:                                                                                                   
        logger.info(f"error executing command!:\n{cmd}")
        logger.info(grepexc.output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="/checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/data.tsv", type=Path)
    parser.add_argument("--emov-km", required=True, type=Path)
    parser.add_argument("--km", nargs='+', required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dict", default="/checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/fairseq.dict.txt")
    parser.add_argument("--manifests-dir", type=Path, default="/checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz")
    args = parser.parse_args()

    manifests_dir = args.manifests_dir
    date = datetime.now().strftime('%d%m%y')
    outdir = manifests_dir / f"{date}"

    # verify input and create folders
    all_kms = args.km + [args.emov_km]
    assert verify_files_exist(all_kms), "make sure the km dir contains: train-clean-all.km, blizzard2013.km, data.km"
    for codes in all_kms:
        assert verify_dict_size(codes, args.dict), "dict argument doesn't match the vocabulary of the km file!"
    assert not outdir.exists(), "data dir already exists!"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("generating denoising split (emov)")
    run_cmd(f"python preprocess/split_km_tsv.py {args.tsv} {args.emov_km} --destdir {outdir}/denoising/emov -sh --seed {args.seed}")
    for codes in args.km:
        codes_name = os.path.basename(codes)
        run_cmd(f"python preprocess/split_km.py {codes} --destdir {outdir}/denoising/{codes_name} -sh --seed {args.seed}")

    logger.info("generating translation split")
    run_cmd(f"python preprocess/split_emov_km_tsv_by_uttid.py {args.tsv} {args.emov_km} --destdir {outdir}/translation --seed {args.seed}")

    emov_code_name = os.path.basename(args.emov_km)
    logger.info("generating hifigan split")
    run_cmd(
        f"mkdir -p {outdir}/hifigan &&"
        f"python preprocess/build_hifigan_manifest.py --km_type hubert --tsv {outdir}/denoising/emov/train.tsv --km {outdir}/denoising/emov/train.km > {outdir}/hifigan/train.txt &&"
        f"python preprocess/build_hifigan_manifest.py --km_type hubert --tsv {outdir}/denoising/emov/valid.tsv --km {outdir}/denoising/emov/valid.km > {outdir}/hifigan/valid.txt &&"
        f"python preprocess/build_hifigan_manifest.py --km_type hubert --tsv {outdir}/denoising/emov/test.tsv --km {outdir}/denoising/emov/test.km > {outdir}/hifigan/test.txt"
    )

    logger.info("generating fairseq manifests")
    run_cmd(f"python preprocess/build_translation_manifests.py {outdir} {outdir}/fairseq-data -dd -cs --dict {args.dict}")

    logger.info(f"finished processing data at:\n{outdir}")


if __name__ == "__main__":
    main()
