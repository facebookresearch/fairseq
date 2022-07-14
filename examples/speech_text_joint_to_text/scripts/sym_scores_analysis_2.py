#!/usr/bin/python3
#
#
#
import argparse
import math
import os
import pickle
import logging

import spacy


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))
ET = {"PERSON", "GPE", "LOC"}


def get_entities(fp):
    full_entities = []
    while True:
        ln = fp.readline().strip()
        if ln == "":
            break
        items = ln.split("\t")
        if items[2] != "O":
            entity_type = items[2].split("-")[1]
            entity_pos = items[2].split("-")[0]
            if entity_pos == "B":
                full_entities.append(([items[1]], entity_type))
            elif entity_pos == "I":
                full_entities[-1][0].append(items[1])
            else:
                raise ValueError("Unrecognized position {} in \"{}\"".format(entity_pos, ln))
    return full_entities


def load_entities(tsv_reference, num_lines):
    entities_list = {et: [] for et in ET}
    with open(tsv_reference) as r_f:
        for _ in range(num_lines):
            es = get_entities(r_f)
            for et in ET:
                entities_list[et].append([" ".join(e) for e, t in es if t == et])
    return entities_list


def normalize_sym_scores(syms):
    sums = [0.0] * len(syms[0])
    square_sums = [0.0] * len(syms[0])
    cnt = 0
    for k in syms.keys():
        cnt += 1
        for i in range(len(syms[k])):
            sums[i] += syms[k][i]
            square_sums[i] += syms[k][i] * syms[k][i]
    mean = [s / cnt for s in sums]
    stddev = [math.sqrt(square_sums[i] / cnt - mean[i] * mean[i]) for i in range(len(square_sums))]
    normalized_syms = {}
    for k in syms.keys():
        normalized_syms[k] = [(syms[k][i] - mean[i]) / stddev[i] for i in range(len(syms[k]))]
    return normalized_syms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv-ref', required=True, type=str, metavar='REFERENCE',
                        help='TSV with NE and terms definition file.')
    parser.add_argument('--sym-scores', required=True, type=str,)
    parser.add_argument('--outputfig', required=True, type=str,)
    parser.add_argument("--list-candidates", type=str, required=True)
    parser.add_argument('--lang', required=True, type=str, metavar='LANG',
                        help='Target language.')

    args = parser.parse_args()

    LANG_MAP = {
        "en": "en_core_web_lg",
        "es": "es_core_news_lg",
        "fr": "fr_core_news_lg",
        "it": "it_core_news_lg"}

    nlp = spacy.load(LANG_MAP[args.lang], disable=['parser', 'ner'])

    # Load candidates
    logger.info('loading candidate list from {}'.format(args.list_candidates))
    with open(args.list_candidates) as f:
        candidate_list = [" ".join(str(tok) for tok in nlp(l.strip().replace(".", " ."))) for l in f]
    logger.info('loading sym_scores from {}'.format(args.sym_scores))
    with open(args.sym_scores, 'rb') as ps_f:
        symilarities = pickle.load(ps_f)  # normalize_sym_scores(pickle.load(ps_f))
    num_lines = len(symilarities)
    logger.info('loading entities')
    entities = load_entities(args.tsv_ref, num_lines)
    thresholds = []
    avg_retrieved = []
    recalls_1w = {et: [] for et in ET}
    recalls_more1 = {et: [] for et in ET}
    for threshold in [1.55 - 0.01 * x for x in range(40)]:
        logger.info(f'computing threshold {threshold}')
        all_reference_1w = {et: 0 for et in ET}
        all_reference_more1 = {et: 0 for et in ET}
        correctly_retrieved_1w = {et: 0 for et in ET}
        correctly_retrieved_more1 = {et: 0 for et in ET}
        total_found = 0.
        for row_idx in range(num_lines):
            found = set(candidate_list[i] for i in range(len(candidate_list)) if symilarities[row_idx][i] > threshold)
            total_found += len(found)
            for et in ET:
                to_find = set(entities[et][row_idx])
                corr_found = to_find.intersection(found)
                correctly_retrieved_more1[et] += sum(1 for ee in corr_found if len(ee.split(" ")) > 1)
                correctly_retrieved_1w[et] += sum(1 for ee in corr_found if len(ee.split(" ")) == 1)
                all_reference_1w[et] += sum(1 for ee in to_find if len(ee.split(" ")) == 1)
                all_reference_more1[et] += sum(1 for ee in to_find if len(ee.split(" ")) > 1)

        for et in ET:
            recalls_1w[et].append(float(correctly_retrieved_1w[et]) / float(all_reference_1w[et]))
            recalls_more1[et].append(float(correctly_retrieved_more1[et]) / float(all_reference_more1[et]))
        avg_retrieved.append(total_found / num_lines)
    import matplotlib.pyplot as plt
    for et in sorted(ET):
        plt.plot(avg_retrieved, recalls_1w[et], label=et + " =1word")
        plt.plot(avg_retrieved, recalls_more1[et], label=et + " >1word")
    plt.legend()
    plt.savefig(args.outputfig.replace('.png', '_bywords.png'))
