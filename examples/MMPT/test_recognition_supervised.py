import random
import statistics

from tqdm import tqdm
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score


def read_file(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
        return lines

seed = 42
random.seed(42)


# embedding_dir = '/shares/iict-sp2.ebling.cl.uzh/zifjia/fairseq/examples/MMPT/runs/retri/baseline_sp_b768/eval'
embedding_dir = '/shares/iict-sp2.ebling.cl.uzh/zifjia/fairseq/examples/MMPT/runs/retri_v1_1/baseline_anonym/eval'
datasets = ['asl_signs', 'asl_citizen', 'sem_lex']
# datasets = ['sem_lex']
top_n = [1, 5, 10]

for dataset in datasets:
    print(f'Evaluating {dataset} ...')

    embedding_dir_train = f'{embedding_dir}/{dataset}_train'
    embedding_dir_test = f'{embedding_dir}/{dataset}_test'

    test_text = read_file(f'{embedding_dir_test}/texts.txt')
    test_embeddings = np.load(f'{embedding_dir_test}/video_embeddings.npy')
    print('test set:', test_embeddings.shape)
    print('test class:', len(set(test_text)))

    train_text = read_file(f'{embedding_dir_train}/texts.txt')
    train_embeddings = np.load(f'{embedding_dir_train}/video_embeddings.npy')
    # train_text = test_text[:100]
    # train_embeddings = test_embeddings[:100]
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

    # scaler = StandardScaler().fit(train_embeddings)
    # X_train = scaler.transform(train_embeddings)
    # X_test = scaler.transform(test_embeddings)

    X_train = train_embeddings
    X_test = test_embeddings

    clf = LogisticRegression(verbose=True, random_state=seed, max_iter=100)
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)
    print(y_score.shape)

    # TODO: optimize for Sem-lex test set (smaller than training)

    for n in top_n:
        score = top_k_accuracy_score(y_test, y_score, labels=range(len(lb.classes_)), k=n)
        print(f'Top {n}:', score)

    y_argsort = np.argsort(-y_score)
    mr = []
    for indice, index in zip(y_argsort, y_test):
        mr.append(list(indice).index(index))
    mr = statistics.median(mr) + 1
    print('Median R:', mr)
        