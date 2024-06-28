import math
import random
import statistics
from collections import defaultdict

from tqdm import tqdm
import numpy as np


def read_file(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
        return lines

def flatten(xss):
    return [x for xs in xss for x in xs]

seed = 42
random.seed(seed)
np.random.seed(seed)


embedding_dir = '/shares/iict-sp2.ebling.cl.uzh/zifjia/fairseq/examples/MMPT/runs/retri_v1_1/baseline_pre/eval_iconicity'
embedding_dir_test = f'{embedding_dir}/test'

test_text = read_file(f'{embedding_dir_test}/texts.txt')
test_text = [t.split(' ')[-1] for t in test_text]
test_embeddings = np.load(f'{embedding_dir_test}/video_embeddings.npy')

ids_grouped = defaultdict(list)
for i, text in enumerate(test_text):
    ids_grouped[text].append(i)

std_by_text = []
for text, ids in ids_grouped.items():
    embeddings = test_embeddings[ids]
    std = np.std(embeddings, axis=0).sum()
    std_by_text.append((text, std))

std_by_text = sorted(std_by_text, key=lambda x: x[1])

for line in std_by_text:
    print(line)