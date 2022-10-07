#!/usr/bin/python3
#
#
#
import argparse
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
        symilarities = pickle.load(ps_f)
    num_lines = len(symilarities)
    logger.info('loading entities')
    entities = load_entities(args.tsv_ref, num_lines)
    thresholds = []
    avg_retrieved = []
    recalls = {et: [] for et in ET}
    for threshold in [0.9 - 0.02 * x for x in range(20)]:
        logger.info(f'computing threshold {threshold}')
        all_reference = {et: 0 for et in ET}
        correctly_retrieved = {et: 0 for et in ET}
        total_found = 0.
        for row_idx in range(num_lines):
            found = set(candidate_list[i] for i in range(len(candidate_list)) if symilarities[row_idx][i] > threshold)
            total_found += len(found)
            for et in ET:
                to_find = set(entities[et][row_idx])
                correctly_retrieved[et] += len(to_find.intersection(found))
                all_reference[et] += len(to_find)
                if len(to_find) > 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"At index {row_idx}, for {et} found: {to_find.intersection(found)} ; missed {to_find.difference(found)}")

                    def get_score(m):
                        if m in candidate_list:
                            return symilarities[row_idx][candidate_list.index(m)]
                        return 0

                    missed_scores = {m: get_score(m) for m in to_find.difference(found)}
                    logger.debug(f"Missed items scores : {missed_scores}")
        for et in ET:
            recalls[et].append(float(correctly_retrieved[et]) / float(all_reference[et]))
        avg_retrieved.append(total_found / num_lines)
        thresholds.append(threshold)
    for i, th in enumerate(thresholds):
        recs = {et: recalls[et][i] for et in sorted(ET)}
        logger.info(f"Recalls at {th} retrieving {avg_retrieved[i]} NEs: {recs}")
    import matplotlib.pyplot as plt
    for et in sorted(ET):
        plt.plot(thresholds, recalls[et], label=et)
    plt.legend()
    plt.savefig(args.outputfig)
    plt.clf()
    for et in sorted(ET):
        plt.plot(avg_retrieved, recalls[et], label=et)
    plt.legend()
    plt.savefig(args.outputfig.replace('.png', '_avgretr.png'))
