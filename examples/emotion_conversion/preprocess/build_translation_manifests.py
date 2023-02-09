from glob import glob
import argparse
from collections import defaultdict, Counter
from itertools import combinations, product, groupby
from pathlib import Path
import os
from sklearn.utils import shuffle
import numpy as np
import random
from shutil import copy
from subprocess import check_call

np.random.seed(42)
random.seed(42)


def get_fname(s):
    return s.split("\t")[0]

def get_emotion(s):
    return get_fname(s).split("_")[0].split("/")[1].lower()

def get_utt_id(s):
    return get_fname(s).split(".")[0].split("_")[-1]

def dedup(seq):
    """ >> remove_repetitions("1 2 2 3 100 2 2 1")
    '1 2 3 100 2 1' """
    seq = seq.strip().split(" ")
    result = seq[:1]
    reps = []
    rep_counter = 1
    for k in seq[1:]:
        if k != result[-1]:
            result += [k]
            reps += [rep_counter]
            rep_counter = 1
        else:
            rep_counter += 1
    reps += [rep_counter]
    assert len(reps) == len(result) and sum(reps) == len(seq)
    return " ".join(result) + "\n" #, reps

def remove_under_k(seq, k):
    """ remove tokens that repeat less then k times in a row
    >> remove_under_k("a a a a b c c c", 1) ==> a a a a c c c """
    seq = seq.strip().split(" ")
    result = []

    freqs = [(k,len(list(g))) for k, g in groupby(seq)]
    for c, f in freqs:
        if f > k:
            result += [c for _ in range(f)]
    return " ".join(result) + "\n" #, reps


def call(cmd):
    print(cmd)
    check_call(cmd, shell=True)


def denoising_preprocess(path, lang, dict):
    bin = 'fairseq-preprocess'
    cmd = [
        bin,
        f'--trainpref {path}/train.{lang} --validpref {path}/valid.{lang} --testpref {path}/test.{lang}',
        f'--destdir {path}/tokenized/{lang}',
        '--only-source',
        '--task multilingual_denoising',
        '--workers 40',
    ]
    if dict != "":
        cmd += [f'--srcdict {dict}']
    cmd = " ".join(cmd)
    call(cmd)


def translation_preprocess(path, src_lang, trg_lang, dict, only_train=False):
    bin = 'fairseq-preprocess'
    cmd = [
        bin,
        f'--source-lang {src_lang} --target-lang {trg_lang}',
        f'--trainpref {path}/train',
        f'--destdir {path}/tokenized',
        '--workers 40',
    ]
    if not only_train:
        cmd += [f'--validpref {path}/valid --testpref {path}/test']
    if dict != "":
        cmd += [
            f'--srcdict {dict}',
            f'--tgtdict {dict}',
        ]
    cmd = " ".join(cmd)
    call(cmd)


def load_tsv_km(tsv_path, km_path):
    assert tsv_path.exists() and km_path.exists()
    tsv_lines = open(tsv_path, "r").readlines()
    root, tsv_lines = tsv_lines[0], tsv_lines[1:]
    km_lines = open(km_path, "r").readlines()
    assert len(tsv_lines) == len(km_lines), ".tsv and .km should be the same length!"
    return root, tsv_lines, km_lines


def main():
    desc = """
    this script takes as input .tsv and .km files for EMOV dataset, and a pairs of emotions.
    it generates parallel .tsv and .km files for these emotions. for exmaple:
    ❯ python build_emov_translation_manifests.py \
            /checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/train.tsv \
            /checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/emov_16khz_km_100/train.km \
            ~/tmp/emov_pairs \
            --src-emotion amused --trg-emotion neutral \
            --dedup --shuffle --cross-speaker --dry-run
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("data", type=Path, help="path to a dir containing .tsv and .km files containing emov dataset")
    parser.add_argument("output_path", type=Path, help="output directory with the manifests will be created")
    parser.add_argument("-cs", "--cross-speaker", action='store_true', help="if set then translation will occur also between speakers, meaning the same sentence can be translated between different speakers (default: false)")
    parser.add_argument("-dd", "--dedup", action='store_true', help="remove repeated tokens (example: 'aaabc=>abc')")
    parser.add_argument("-sh", "--shuffle", action='store_true', help="shuffle the data")
    parser.add_argument("-ae", "--autoencode", action='store_true', help="include training pairs from the same emotion (this includes examples of the same sentence uttered by different people and examples where the src and trg are the exact same seq)")
    parser.add_argument("-dr", "--dry-run", action='store_true', help="don't write anything to disk")
    parser.add_argument("-zs", "--zero-shot", action='store_true', help="if true, the denoising task will train on the same splits as the translation task (split by utterance id). if false, the denoising task will train on randomly sampled splits (not split by utterance id)")
    parser.add_argument("--km-ext", default="km", help="")
    parser.add_argument("--dict", default="/checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/fairseq.dict.txt", help="")
    args = parser.parse_args()
    SPEAKERS = ["bea", "jenie", "josh", "sam", "SAME"]
    EMOTIONS = ['neutral', 'amused', 'angry', 'disgusted', 'sleepy']

    suffix = ""
    if args.cross_speaker: suffix += "_cross-speaker"
    if args.dedup: suffix += "_dedup"
    translation_suffix = ""
    if args.autoencode: translation_suffix += "_autoencode"
    denoising_suffix = ""
    denoising_suffix += "_zeroshot" if args.zero_shot else "_nonzeroshot"

    translation_dir = Path(args.output_path) / ("emov_multilingual_translation" + suffix + translation_suffix)
    os.makedirs(translation_dir, exist_ok=True)
    denoising_dir = Path(args.output_path) / ("emov_multilingual_denoising" + suffix + denoising_suffix)
    os.makedirs(denoising_dir, exist_ok=True)

    denoising_data = [p.name for p in (args.data / "denoising").glob("*") if "emov" not in p.name]

    for split in ["train", "valid", "test"]:
        root, tsv_lines, km_lines = load_tsv_km(
            tsv_path = args.data / "denoising" / "emov" / f"{split}.tsv",
            km_path = args.data / "denoising" / "emov" / f"{split}.{args.km_ext}"
        )

        # generate data for the multilingual denoising task
        for EMOTION in EMOTIONS:
            print("---")
            print(split)
            print(f"denoising: {EMOTION}")
            emotion_tsv, emotion_km = [], []
            for tsv_line, km_line in zip(tsv_lines, km_lines):
                if EMOTION.lower() in tsv_line.lower():
                    km_line = km_line if not args.dedup else dedup(km_line)
                    emotion_tsv.append(tsv_line)
                    emotion_km.append(km_line)
            print(f"{len(emotion_km)} samples")
            open(denoising_dir / f"files.{split}.{EMOTION}", "w").writelines([root] + emotion_tsv)
            open(denoising_dir / f"{split}.{EMOTION}", "w").writelines(emotion_km)

        for data in denoising_data:
            with open(args.data / "denoising" / data / f"{split}.{args.km_ext}", "r") as f1:
                with open(denoising_dir / f"{split}.{data}", "w") as f2:
                    f2.writelines([l if not args.dedup else dedup(l) for l in f1.readlines()])

        # start of translation preprocessing
        root, tsv_lines, km_lines = load_tsv_km(
            tsv_path = args.data / "translation" / f"{split}.tsv",
            km_path = args.data / "translation" / f"{split}.{args.km_ext}"
        )

        # generate data for the multilingual translation task
        for SRC_EMOTION in EMOTIONS:
            TRG_EMOTIONS = EMOTIONS if args.autoencode else set(EMOTIONS) - set([SRC_EMOTION])
            for TRG_EMOTION in TRG_EMOTIONS:
                # when translating back to the same emotion - we dont want these emotion
                # pairs to be part of the validation/test sets (because its not really emotion conversino)
                #  if SRC_EMOTION == TRG_EMOTION and split in ["valid", "test"]: continue
                print("---")
                print(split)
                print(f"src emotions: {SRC_EMOTION}\ntrg emotions: {TRG_EMOTION}")

                # create a dictionary with the following structure:
                # output[SPEAKER][UTT_ID] = list with indexes of line from the tsv file
                # that match the speaker and utterance id. for exmaple:
                # output = {'sam': {'0493': [875, 1608, 1822], ...}, ...}
                # meaning, for speaker 'sam', utterance id '0493', the indexes in tsv_lines
                # are 875, 1608, 1822
                spkr2utts = defaultdict(lambda: defaultdict(list))
                for i, tsv_line in enumerate(tsv_lines):
                    speaker = tsv_line.split("/")[0]
                    if args.cross_speaker: speaker = "SAME"
                    assert speaker in SPEAKERS, "unknown speaker! make sure the .tsv contains EMOV data"
                    utt_id = get_utt_id(tsv_line)
                    spkr2utts[speaker][utt_id].append(i)

                # create a tsv and km files with all the combinations for translation
                src_tsv, trg_tsv, src_km, trg_km = [], [], [], []
                for speaker, utt_ids in spkr2utts.items():
                    for utt_id, indices in utt_ids.items():
                        # generate all pairs
                        pairs = [(x,y) for x in indices for y in indices]
                        # self-translation 
                        if SRC_EMOTION == TRG_EMOTION:
                            pairs = [(x,y) for (x,y) in pairs if x == y]
                        # filter according to src and trg emotions
                        pairs = [(x,y) for (x,y) in pairs 
                                if get_emotion(tsv_lines[x]) == SRC_EMOTION and get_emotion(tsv_lines[y]) == TRG_EMOTION]

                        for idx1, idx2 in pairs:
                            assert get_utt_id(tsv_lines[idx1]) == get_utt_id(tsv_lines[idx2])
                            src_tsv.append(tsv_lines[idx1])
                            trg_tsv.append(tsv_lines[idx2])
                            km_line_idx1 = km_lines[idx1]
                            km_line_idx2 = km_lines[idx2]
                            km_line_idx1 = km_line_idx1 if not args.dedup else dedup(km_line_idx1)
                            km_line_idx2 = km_line_idx2 if not args.dedup else dedup(km_line_idx2)
                            src_km.append(km_line_idx1)
                            trg_km.append(km_line_idx2)
                assert len(src_tsv) == len(trg_tsv) == len(src_km) == len(trg_km)
                print(f"{len(src_tsv)} pairs")

                if len(src_tsv) == 0:
                    raise Exception("ERROR: generated 0 pairs!")

                if args.dry_run: continue

                # create files
                os.makedirs(translation_dir / f"{SRC_EMOTION}-{TRG_EMOTION}", exist_ok=True)
                open(translation_dir / f"{SRC_EMOTION}-{TRG_EMOTION}" / f"files.{split}.{SRC_EMOTION}", "w").writelines([root] + src_tsv)
                open(translation_dir / f"{SRC_EMOTION}-{TRG_EMOTION}" / f"files.{split}.{TRG_EMOTION}", "w").writelines([root] + trg_tsv)
                open(translation_dir / f"{SRC_EMOTION}-{TRG_EMOTION}" / f"{split}.{SRC_EMOTION}", "w").writelines(src_km)
                open(translation_dir / f"{SRC_EMOTION}-{TRG_EMOTION}" / f"{split}.{TRG_EMOTION}", "w").writelines(trg_km)

        
    # fairseq-preprocess the denoising data
    for EMOTION in EMOTIONS + denoising_data:
        denoising_preprocess(denoising_dir, EMOTION, args.dict)
    os.system(f"cp {args.dict} {denoising_dir}/tokenized/dict.txt")

    # fairseq-preprocess the translation data
    os.makedirs(translation_dir / "tokenized", exist_ok=True)
    for SRC_EMOTION in EMOTIONS:
        TRG_EMOTIONS = EMOTIONS if args.autoencode else set(EMOTIONS) - set([SRC_EMOTION])
        for TRG_EMOTION in TRG_EMOTIONS:
            translation_preprocess(translation_dir / f"{SRC_EMOTION}-{TRG_EMOTION}", SRC_EMOTION, TRG_EMOTION, args.dict)#, only_train=SRC_EMOTION==TRG_EMOTION)
    os.system(f"cp -rf {translation_dir}/**/tokenized/* {translation_dir}/tokenized")

if __name__ == "__main__":
    main()
