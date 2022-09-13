#!python3
import sqlite3 as sl
import argparse
import os
import pickle
import logging
import spacy

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))
ET = {"PERSON", "GPE", "LOC", "ORG"}


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
        for i in range(num_lines):
            es = get_entities(r_f)
            entities_list.append(set([" ".join(e) for e, t in es if t in ET]))
    return entities_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv-ref', required=True, type=str, metavar='REFERENCE',
                        help='TSV with NE and terms definition file.')
    parser.add_argument('--sym-scores', required=True, type=str,)
    parser.add_argument("--list-candidates", type=str, required=True)
    parser.add_argument('--lang', required=True, type=str, metavar='LANG',
                        help='Target language.')

    args = parser.parse_args()

    LANG_MAP = {
        "en": "en_core_web_lg",
        "es": "es_core_news_lg",
        "fr": "fr_core_news_lg",
        "it": "it_core_news_lg"}

    nlp = spacy.load(LANG_MAP["en"], disable=['parser', 'ner'])

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
    con = sl.connect('ne_testonly.db')
    with con:
        con.execute(f"create table if not exists scores_{args.lang}(s_id INT, ne_id INT, is_ne_present INT, score DOUBLE, is_ne_retrieved INT, primary key (s_id, ne_id))")

    rows = []
    threshold = 0.86
    for s_id in range(num_lines):
        for ne_id in range(len(candidate_list)):
            retrieved = symilarities[s_id][ne_id] > threshold
            should_be_found = candidate_list[ne_id] in entities[s_id]
            rows.append((s_id, ne_id, int(should_be_found), symilarities[s_id][ne_id], int(retrieved)))
    with con:
        con.executemany(f"insert into scores_{args.lang} (s_id, ne_id, is_ne_present, score, is_ne_retrieved) values(?, ?, ?, ?, ?)", rows)
