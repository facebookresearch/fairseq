import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
import subprocess
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--dump', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument("--format", type=str, choices=["none", "letter"], default="letter")
    parser.add_argument("--extra-infer-args", type=str, default="")
    parser.add_argument('--use_lm', type=int, default=0)
    parser.add_argument('--lm_dir', type=str, default="/checkpoint/yanb/MMS1_models/lm/mms-cclms")
    args = parser.parse_args()

    langs = [d for d in os.listdir(args.dump) if os.path.isdir(os.path.join(args.dump, d))]

    for lang in langs:
        print(f"python examples/mms/asr/scripts/run_single_lang.py --dump {args.dump} --lang {lang} --model {args.model} --dst {args.dst} --use_lm {args.use_lm}")
    