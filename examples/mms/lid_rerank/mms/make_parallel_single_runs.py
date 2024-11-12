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
    parser.add_argument('--fairseq_dir', type=str)
    args = parser.parse_args()

    langs = [d for d in os.listdir(args.dump) if os.path.isdir(os.path.join(args.dump, d))]

    for lang in langs:
        print(f"python mms/run_single_lang.py --dump {os.path.abspath(args.dump)} --lang {lang} --model {os.path.abspath(args.model)} --dst {os.path.abspath(args.dst)} --fairseq_dir {os.path.abspath(args.fairseq_dir)}")
    