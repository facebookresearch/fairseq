import argparse
import csv
import sqlite3 as sl


def main(args):
    dict_entities = {}
    with open(args.dict_tsv) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        for i, line in enumerate(reader):
            dict_entities[i] = line[args.lang]

    # Connecting to the database
    con = sl.connect(args.db_file)
    cursor = con.execute(
        "select s_id, ne_id "
        f"from {args.scores_table} where "
        "ne_id not in (96, 97, 99, 123, 132, 133, 134, 151, 152, 155, 157, 158, 168, 169, 174, 186, 187, 188, 152, 97, 169) "  # no acronyms
        f"and score > {args.score_threshold}")
    entities_retrieved = {}
    for r in cursor.fetchall():
        if r[0] not in entities_retrieved:
            entities_retrieved[r[0]] = []
        entities_retrieved[r[0]].append(dict_entities[r[1]])

    with open(args.test_tsv) as f, open(args.out_tsv, 'w') as out_f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        new_fieldnames = reader.fieldnames
        if 'entities' not in new_fieldnames:
            new_fieldnames.append('entities')
        writer = csv.DictWriter(
            out_f,
            fieldnames=new_fieldnames,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        writer.writeheader()
        for i, line in enumerate(reader):
            if i in entities_retrieved:
                entities = " ; ".join(entities_retrieved[i])
            else:
                entities = ""
            line['entities'] = entities
            writer.writerow(line)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-file", type=str, required=True)
    parser.add_argument("--scores-table", type=str, required=True)
    parser.add_argument("--dict-tsv", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--test-tsv", type=str, required=True)
    parser.add_argument("--out-tsv", type=str, required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()

