import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import sys
import subprocess
import re
import pandas as pd
import random
import pycountry
import ast
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--dir', type=str)
    parser.add_argument('--splits', type=int)
    args = parser.parse_args()

    full_data = []
    for i in range(args.splits):
        split_data = open(args.dir + "/split_" + str(i) + "/predictions.txt", "r").readlines()
        full_data += split_data

    with open(args.dir + "/predictions.txt", "w") as f:
        f.writelines(full_data)