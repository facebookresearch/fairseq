#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import argparse
import glob


DIM = 1024


def compute_dist(source_embs, target_embs, k=5, return_sim_mat=False):
    target_ids = [tid for tid in target_embs]
    source_mat = np.stack(source_embs.values(), axis=0)
    normalized_source_mat = source_mat / np.linalg.norm(source_mat, axis=1, keepdims=True)
    target_mat = np.stack(target_embs.values(), axis=0)
    normalized_target_mat = target_mat / np.linalg.norm(target_mat, axis=1, keepdims=True)
    sim_mat = normalized_source_mat.dot(normalized_target_mat.T)
    if return_sim_mat:
        return sim_mat
    neighbors_map = {}
    for i, sentence_id in enumerate(source_embs):
        idx = np.argsort(sim_mat[i, :])[::-1][:k]
        neighbors_map[sentence_id] = [target_ids[tid] for tid in idx]
    return neighbors_map


def load_embeddings(directory, LANGS):
    sentence_embeddings = {}
    sentence_texts = {}
    for lang in LANGS:
        sentence_embeddings[lang] = {}
        sentence_texts[lang] = {}
        lang_dir = f"{directory}/{lang}"
        embedding_files = glob.glob(f"{lang_dir}/all_avg_pool.{lang}.*")
        for embed_file in embedding_files:
            shard_id = embed_file.split('.')[-1]
            embeddings = np.fromfile(embed_file, dtype=np.float32)
            num_rows = embeddings.shape[0] // DIM
            embeddings = embeddings.reshape((num_rows, DIM))

            with open(f'{lang_dir}/sentences.{lang}.{shard_id}') as sentence_file:
                for idx, line in enumerate(sentence_file):
                    sentence_id, sentence = line.strip().split('\t')
                    sentence_texts[lang][sentence_id] = sentence
                    sentence_embeddings[lang][sentence_id] = embeddings[idx, :]

    return sentence_embeddings, sentence_texts


def compute_accuracy(directory, LANGS):
    sentence_embeddings, sentence_texts = load_embeddings(directory, LANGS)

    top_1_accuracy = {}

    top1_str = " ".join(LANGS) + '\n'
    for source_lang in LANGS:
        top_1_accuracy[source_lang] = {}
        top1_str += f"{source_lang} "
        for target_lang in LANGS:
            top1 = 0
            top5 = 0
            neighbors_map = compute_dist(
                sentence_embeddings[source_lang],
                sentence_embeddings[target_lang])
            for sentence_id, neighbors in neighbors_map.items():
                if sentence_id == neighbors[0]:
                    top1 += 1
                if sentence_id in neighbors[:5]:
                    top5 += 1
            n = len(sentence_embeddings[target_lang])
            top1_str += f"{top1/n} "
        top1_str += "\n"

    print(top1_str)
    print(top1_str, file=open(f"{directory}/accuracy", 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze encoder outputs')
    parser.add_argument('directory',
        help='Source language corpus'
    )
    parser.add_argument('--langs',
        help='List of langs'
    )
    args = parser.parse_args()
    langs = args.langs.split(',')
    compute_accuracy(args.directory, langs)
