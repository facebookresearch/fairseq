import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
import subprocess
import re

mapping = {"cmn":"cmn-script_simplified", "srp":"srp-script_latin", "urd":"urd-script_arabic", "uzb":"uzb-script_latin", "yue":"yue-script_traditional", "aze":"azj-script_latin", "kmr":"kmr-script_latin"}

def reorder_decode(hypos):
    outputs = []
    for hypo in hypos:
        idx = int(re.findall("\(None-(\d+)\)$", hypo)[0])
        hypo = re.sub("\(\S+\)$", "", hypo).strip()
        outputs.append((idx, hypo))
    outputs = sorted(outputs)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--dump', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--fairseq_dir', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--lang', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)
    lang = args.lang
    dst = args.dst + "/" + lang
    if not os.path.exists(dst):
        os.makedirs(dst)
    dump = args.dump + "/" + lang
    if lang in mapping:
        lang_code = mapping[lang]
    else:
        lang_code = lang

    cmd = f"""
    cd {args.fairseq_dir}/ &&\
    PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m --config-dir examples/mms/asr/config/ --config-name infer_common decoding.type=viterbi dataset.max_tokens=1440000 distributed_training.distributed_world_size=1 "common_eval.path='{args.model}'" task.data={dump} dataset.gen_subset="{lang_code}:test" common_eval.post_process=letter decoding.results_path={dst} &&\
    cd -
    """

    print(cmd, file=sys.stderr)
    print(f">>> {lang}", file=sys.stderr)
    try:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,)
        with open(dst + "/hypo.word") as fr, open(dst + "/hypo.word.reord", "w") as fw:
            hypos = fr.readlines()
            outputs = reorder_decode(hypos)
            fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
        with open(dst + "/asr_score") as fr, open(dst + "/asr_score.reord", "w") as fw:
            hypos = fr.readlines()
            outputs = reorder_decode(hypos)
            fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
    except:
        print(f"Something went wrong with {lang}. If {lang} is not supported by the ASR model, then this is expected and OK. If it is supported, then something else has gone wrong unexpectedly.", file=sys.stderr)
        with open(dst + "/hypo.word.reord", "w") as fw:
            fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))
        with open(dst + "/asr_score.reord", "w") as fw:
            fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))