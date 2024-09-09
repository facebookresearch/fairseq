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
    parser.add_argument('--dst', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument("--format", type=str, choices=["none", "letter"], default="letter")
    parser.add_argument("--extra-infer-args", type=str, default="")
    parser.add_argument('--use_lm', type=int, default=0)
    parser.add_argument('--lm_dir', type=str, default="/checkpoint/yanb/MMS1_models/lm/mms-cclms/")
    args = parser.parse_args()

    lang = args.lang
    dst = args.dst + "/" + lang
    if not os.path.exists(dst):
        os.makedirs(dst)
    dump = args.dump + "/" + lang
    if lang in mapping:
        lang_code = mapping[lang]
    else:
        lang_code = lang

    # LM
    if args.use_lm != 0:
        config_all = json.load(open(args.lm_dir + "/mms-1b-all/decoding_config.json", "r"))
        if lang_code not in config_all:
            cmd = f"""
            PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m --config-dir examples/mms/asr/config/ --config-name infer_common decoding.type=viterbi dataset.max_tokens=1440000 distributed_training.distributed_world_size=1 "common_eval.path='{args.model}'" task.data={dump} dataset.gen_subset="{lang_code}:test" common_eval.post_process={args.format} decoding.results_path={dst} {args.extra_infer_args}
            """
        else:
            config = config_all[lang_code]

            char_lms = ["cmn", "yue", "jpn", "tha", "lao", "mya", "khm"]
            if lang in char_lms:
                unitlm=True
                lexicon=""
            else:
                unitlm=False
                lexicon=args.lm_dir+config['lexiconfile']
            #     # lm_file = args.lm_dir + "/lms/" + lang_code + "char_20gram.bin"
            #     cmd = f"""
            #     PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m --config-dir examples/mms/asr/config/ --config-name infer_common decoding.type=kenlm distributed_training.distributed_world_size=1 decoding.unique_wer_file=true decoding.beam=500 decoding.beamsizetoken=50 task.data={dump} "common_eval.path='{args.model}'" decoding.lexicon="" decoding.lmpath="{lm_file}" decoding.results_path={dst} dataset.gen_subset="{lang_code}:test" decoding.lmweight=2.5 decoding.unitlm=True decoding.silweight=0 {args.extra_infer_args}
            #     """
            # else:
                # lm_file = args.lm_dir + "/lms/" + lang_code + "word_5gram.bin"
                # lexicon_file = args.lm_dir + "/mms-1b-all/" + lang_code + "lexicon.txt"
            cmd = f"""
            PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m --config-dir examples/mms/asr/config/ --config-name infer_common decoding.type=kenlm distributed_training.distributed_world_size=1 decoding.unique_wer_file=true decoding.beam=500 decoding.beamsizetoken=50 task.data={dump} "common_eval.path='{args.model}'" decoding.lexicon="{lexicon}" decoding.lmpath="{args.lm_dir+config['lmfile']}" decoding.results_path={dst} dataset.gen_subset="{lang_code}:test" decoding.lmweight={config['lmweight']} decoding.wordscore={config['wordscore']} decoding.silweight={config['silweight']} decoding.unitlm={unitlm} {args.extra_infer_args}
            """
            print("USING LM", file=sys.stderr)
    # no LM
    else:
        cmd = f"""
        PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m --config-dir examples/mms/asr/config/ --config-name infer_common decoding.type=viterbi dataset.max_tokens=1440000 distributed_training.distributed_world_size=1 "common_eval.path='{args.model}'" task.data={dump} dataset.gen_subset="{lang_code}:test" common_eval.post_process={args.format} decoding.results_path={dst} {args.extra_infer_args}
        """

    print(cmd, file=sys.stderr)
    print(f">>> {lang}", file=sys.stderr)
    try:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,)
        with open(dst + "/hypo.word") as fr, open(dst + "/hypo.word.reord", "w") as fw:
            hypos = fr.readlines()
            outputs = reorder_decode(hypos)
            fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
        with open(dst + "/hypo.units") as fr, open(dst + "/hypo.units.reord", "w") as fw:
            hypos = fr.readlines()
            outputs = reorder_decode(hypos)
            fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
        with open(dst + "/hypo.score") as fr, open(dst + "/hypo.score.reord", "w") as fw:
            hypos = fr.readlines()
            outputs = reorder_decode(hypos)
            fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
        with open(dst + "/p") as fr, open(dst + "/p.reord", "w") as fw:
            hypos = fr.readlines()
            outputs = reorder_decode(hypos)
            fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
    except:
        # try without LM
        if args.use_lm != 0:
            cmd = f"""
            PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m --config-dir examples/mms/asr/config/ --config-name infer_common decoding.type=viterbi dataset.max_tokens=1440000 distributed_training.distributed_world_size=1 "common_eval.path='{args.model}'" task.data={dump} dataset.gen_subset="{lang_code}:test" common_eval.post_process={args.format} decoding.results_path={dst} {args.extra_infer_args}
            """
            print(cmd, file=sys.stderr)
            try:
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,)
                with open(dst + "/hypo.word") as fr, open(dst + "/hypo.word.reord", "w") as fw:
                    hypos = fr.readlines()
                    outputs = reorder_decode(hypos)
                    fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
                with open(dst + "/hypo.units") as fr, open(dst + "/hypo.units.reord", "w") as fw:
                    hypos = fr.readlines()
                    outputs = reorder_decode(hypos)
                    fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
                with open(dst + "/hypo.score") as fr, open(dst + "/hypo.score.reord", "w") as fw:
                    hypos = fr.readlines()
                    outputs = reorder_decode(hypos)
                    fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
                with open(dst + "/p") as fr, open(dst + "/p.reord", "w") as fw:
                    hypos = fr.readlines()
                    outputs = reorder_decode(hypos)
                    fw.writelines([re.sub("\(\S+\)$", "", hypo).strip() + "\n" for ii,hypo in outputs])
            except:
                print(f"Something went wrong with {lang}", file=sys.stderr)
                with open(dst + "/hypo.word.reord", "w") as fw:
                    fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))
                with open(dst + "/hypo.units.reord", "w") as fw:
                    fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))
                with open(dst + "/hypo.score.reord", "w") as fw:
                    fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))
                with open(dst + "/p.reord", "w") as fw:
                    fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))
        else:
            print(f"Something went wrong with {lang}", file=sys.stderr)
            with open(dst + "/hypo.word.reord", "w") as fw:
                fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))
            with open(dst + "/hypo.units.reord", "w") as fw:
                fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))
            with open(dst + "/hypo.score.reord", "w") as fw:
                fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))
            with open(dst + "/p.reord", "w") as fw:
                    fw.writelines(["\n"] * len(open(dump+"/ids.txt", "r").readlines()))