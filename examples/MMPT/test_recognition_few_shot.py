import random
import statistics
from collections import defaultdict

from tqdm import tqdm
import numpy as np


def read_file(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
        return lines

random.seed(42)


# embedding_dir = '/shares/iict-sp2.ebling.cl.uzh/zifjia/fairseq/examples/MMPT/runs/retri/baseline_sp_b768/eval'
embedding_dir = '/shares/iict-sp2.ebling.cl.uzh/zifjia/fairseq/examples/MMPT/runs/retri_v1_1/baseline_anonym/eval'
datasets = ['asl_signs', 'asl_citizen', 'sem_lex']
# datasets = ['sem_lex']
number_shots = [1, 5, 10, 100]

for dataset in datasets:
    for number_shot in number_shots:
        print(f'Evaluating {dataset} {number_shot}-shot ...')

        embedding_dir_train = f'{embedding_dir}/{dataset}_train'
        embedding_dir_test = f'{embedding_dir}/{dataset}_test'

        test_text = read_file(f'{embedding_dir_test}/texts.txt')
        test_embeddings = np.load(f'{embedding_dir_test}/video_embeddings.npy')

        train_text = read_file(f'{embedding_dir_train}/texts.txt')
        train_embeddings = np.load(f'{embedding_dir_train}/video_embeddings.npy')
        train_embeddings_grouped = defaultdict(list)

        for i, text in enumerate(train_text):
            train_embeddings_grouped[text].append(i)

        # print(train_embeddings_grouped)
        # print(len(train_embeddings_grouped))
        # exit()

        for text, ids in train_embeddings_grouped.items():
            if len(ids) > number_shot:
                train_embeddings_grouped[text] = random.sample(ids, number_shot)

        hit = 0
        hit_5 = 0
        hit_10 = 0
        mr = []

        # test_text = test_text[:10]
        for i, gold_text in tqdm(enumerate(test_text)):
            test_embedding = test_embeddings[i]
            scores_per_class = []

            for text, ids in train_embeddings_grouped.items():
                candidate_embeddings = train_embeddings[ids]
                scores = np.matmul(np.expand_dims(test_embedding, axis=0), candidate_embeddings.T).squeeze()
                avg_score = np.average(scores)
                scores_per_class.append((text, avg_score))

            scores_per_class = sorted(scores_per_class, key=lambda x: x[1], reverse=True)
            text_candidates = [x[0] for x in scores_per_class]

            # print(gold_text)
            # print(text_candidates)

            if gold_text in text_candidates[:1]:
                hit = hit + 1
            if gold_text in text_candidates[:5]:
                hit_5 = hit_5 + 1
            if gold_text in text_candidates[:10]:
                hit_10 = hit_10 + 1
            if gold_text in text_candidates:
                mr.append(text_candidates.index(gold_text))
            
            
        recall = hit / len(test_text)
        recall_5 = hit_5 / len(test_text)
        recall_10 = hit_10 / len(test_text)
        mr = statistics.median(mr) + 1

        print('recall:', recall)
        print('recall_5:', recall_5)
        print('recall_10:', recall_10)
        print('mr:', mr)