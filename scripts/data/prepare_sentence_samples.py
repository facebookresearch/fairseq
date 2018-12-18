#!/usr/bin/env python3

import argparse
import random


def main(args):
    if args.randomize_next_sentence:
        docs = []
        with open(args.input, 'r') as inp:
            doc = []
            for line in inp:
                line = line.strip()
                if len(line) == 0 and len(doc) > 0:
                    docs.append(doc)
                    doc = []
                elif len(line) > 0:
                    doc.append(line)

        with open(args.output, 'w') as out, open(args.output + '.lbl', 'w') as lbl:
            assert len(docs) > 1
            for i, doc in enumerate(docs):
                s1 = doc[0]
                j = 1
                while j <= len(doc):
                    if random.random() >= 0.5 or j == len(doc):
                        r = i
                        while r == i:
                            r = random.randint(0, len(docs) - 1)
                        k = random.randint(0, len(docs[r]) - 1)
                        s2 = docs[r][k]
                        print(0, file=lbl)
                    else:
                        s2 = doc[j]
                        j += 1
                        print(1, file=lbl)
                    sample = f'{s1} {args.sep} {s2}'
                    print(sample, file=out)
                    if j < len(doc):
                        s1 = doc[j]
                    j += 1
    else:
        with open(args.input, 'r') as inp, open(args.output, 'w') as out:
            prev_sent = None
            for line in inp:
                line = line.strip()
                if len(line) == 0:
                    if args.keep_single and prev_sent is not None:
                        sample = f'{prev_sent}'
                        print(sample, file=out)
                    prev_sent = None
                elif prev_sent is None:
                    prev_sent = line
                else:
                    sample = f'{prev_sent} {args.sep} {line}'
                    print(sample, file=out)
                    prev_sent = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        metavar='FILE',
        help='files to process.',
    )
    parser.add_argument(
        '--output',
        required=True,
        metavar='FILE',
        help='Path for output',
    )
    parser.add_argument(
        '--sep',
        default='<SEP>',
        help='separator token',
    )
    parser.add_argument(
        '--keep-single',
        action='store_true',
        help='if set, keeps single example sentences also (e.g. for uneven length docs)'
    )
    parser.add_argument(
        '--randomize-next-sentence',
        action='store_true',
        help='if set, next sentence has a 50% chance of being from a different document, and the results are stored in .lbl file'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        help='if set, sequences are constructed by taking up to this many tokens. when the number of tokens exceeds this length, they are truncated. sentences are assigned to first or second segment randomly'
    )
    args = parser.parse_args()
    print(args)
    main(args)
