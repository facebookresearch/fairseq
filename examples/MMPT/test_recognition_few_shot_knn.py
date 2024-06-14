import math
import random
import statistics
from collections import defaultdict

from tqdm import tqdm
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import top_k_accuracy_score


def read_file(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
        return lines

def flatten(xss):
    return [x for xs in xss for x in xs]

seed = 42
random.seed(seed)
np.random.seed(seed)


# embedding_dir = '/shares/iict-sp2.ebling.cl.uzh/zifjia/fairseq/examples/MMPT/runs/retri/baseline_sp_b768/eval'
# embedding_dir = '/shares/iict-sp2.ebling.cl.uzh/zifjia/fairseq/examples/MMPT/runs/retri_v1_1/baseline_anonym/eval'
datasets = ['asl_signs', 'asl_citizen', 'sem_lex']
# datasets = ['sem_lex']
number_shots = [1, 5, 10, 100]
top_n = [1, 5, 10]

for dataset in datasets:
    if dataset == 'asl_signs':
        embedding_dir = '/shares/iict-sp2.ebling.cl.uzh/zifjia/fairseq/examples/MMPT/runs/retri_v1_1/baseline_temporal/eval_anonym_full'
    else:
        embedding_dir = '/shares/iict-sp2.ebling.cl.uzh/zifjia/fairseq/examples/MMPT/runs/retri_v1_1/baseline_temporal/eval_flip_full'

    for number_shot in number_shots:
        print(f'Evaluating {dataset} {number_shot}-shot ...')

        embedding_dir_train = f'{embedding_dir}/{dataset}_train'
        embedding_dir_test = f'{embedding_dir}/{dataset}_test'

        test_text = read_file(f'{embedding_dir_test}/texts.txt')
        test_embeddings = np.load(f'{embedding_dir_test}/video_embeddings.npy')

        train_text = read_file(f'{embedding_dir_train}/texts.txt')
        train_embeddings = np.load(f'{embedding_dir_train}/video_embeddings.npy')
        train_ids_grouped = defaultdict(list)

        for i, text in enumerate(train_text):
            if text in test_text:
                train_ids_grouped[text].append(i)

        # print(train_ids_grouped)
        # print(len(train_ids_grouped))
        # exit()

        for text, ids in train_ids_grouped.items():
            if len(ids) > number_shot:
                train_ids_grouped[text] = random.sample(ids, number_shot)

        indices = flatten(train_ids_grouped.values())
        train_text = [train_text[i] for i in indices]
        train_embeddings = train_embeddings[indices]

        print('test set:', test_embeddings.shape)
        print('test class:', len(set(test_text)))

        print('train set:', train_embeddings.shape)
        print('train class:', len(set(train_text)))

        print('Filter test set with known labels')
        ids = []
        test_text_filtered = []
        for i, text in enumerate(test_text):
            if text in train_text:
                ids.append(i)
                test_text_filtered.append(text)
        test_text = test_text_filtered
        test_embeddings = test_embeddings[ids]
        print('test set:', test_embeddings.shape)

        lb = LabelEncoder()
        y_train = lb.fit_transform(train_text)
        y_test = lb.transform(test_text)

        y_train_labels = list(sorted(set(y_train)))
        y_test_labels = list(sorted(set(y_test)))

        # print('test class:', y_train_labels)
        # print('test class:', y_test_labels)
        print('class in training set:', len(y_train_labels))
        print('class in test set:', len(y_test_labels))

        X_train = train_embeddings
        X_test = test_embeddings

        # n_neighbors = len(y_test_labels)
        n_neighbors = round(math.sqrt(X_train.shape[0]))

        clf = Pipeline(
            steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))]
        )
        clf.fit(X_train, y_train)
        # clf.fit(np.random.rand(len(y_train_labels), X_train.shape[1]), y_train_labels)

        y_score = clf.predict_proba(X_test)
        print(y_score.shape)

        for n in top_n:
            score = top_k_accuracy_score(y_test, y_score, labels=y_train_labels, k=n)
            print(f'Top {n}:', score)

        y_argsort = np.argsort(-y_score)
        mr = []
        for indice, index in zip(y_argsort, y_test):
            mr.append(list(indice).index(index))
        mr = statistics.median(mr) + 1
        print('Median R:', mr)