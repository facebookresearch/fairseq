#!/usr/bin/python3
#
#
#
import argparse
import pickle

import spacy


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
    entities_list = []
    with open(tsv_reference) as r_f:
        for _ in range(num_lines):
            entities_list.append([" ".join(e) for e, t in get_entities(r_f) if t in {"PERSON", "GPE", "LOCATION"}])
    assert len(entities_list) == num_lines
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
    print('loading candidate list from {}'.format(args.list_candidates))
    with open(args.list_candidates) as f:
        candidate_list = [" ".join(str(tok) for tok in nlp(l.strip())) for l in f]
    print('loading sym_scores from {}'.format(args.sym_scores))
    with open(args.sym_scores, 'rb') as ps_f:
        symilarities = pickle.load(ps_f)
    num_lines = len(symilarities)
    print('loading entities')
    entities = load_entities(args.tsv_ref, num_lines)
    precisions = []
    recalls = []
    for threshold in [.8 - 0.05 * x for x in range(15)]:
        print(f'computing threshold {threshold}')
        all_reference = 0
        all_retrieved = 0
        correctly_retrieved = 0
        for row_idx in range(num_lines):
            to_find = set(entities[row_idx])
            if len(to_find) > 0:
                import pdb ; pdb.set_trace()
            found = set(candidate_list[i] for i in range(len(candidate_list)) if symilarities[row_idx][i] > threshold)
            all_reference += len(to_find)
            all_retrieved += len(found)
            correctly_retrieved += len(to_find.intersection(found))
        recall = float(correctly_retrieved) / float(all_reference)
        precision = float(correctly_retrieved) / float(all_retrieved) if all_retrieved > 0 else 0.0
        print(f"precision: {precision} ; recall {recall}")
        recalls.append(recall)
        precisions.append(precision)
    import matplotlib.pyplot as plt 
    plt.plot(precisions, recalls)
    plt.savefig(args.outputfig)
