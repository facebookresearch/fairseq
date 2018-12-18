#!/usr/bin/env python3

import argparse
import os
import re

import nltk

remove_startswith = [
    'Published by',
    'Table of Contents',
    'Copyright',
    'Smashwords',
    'All rights reserved',
    'ISBN',
    '\uFEFF'
]
remove_regex = [
    re.compile(r'^Chapter \d+.*'),  # chapter
    re.compile(r'^By \w+ \w+$'),  # author
    re.compile(r'^\d+ ?\S+$'),  # chapter headings
]


def normalize(sent, num):
    def count(s, c):
        return sum(x == c for x in s)

    # check for header / table of contents / etc only in the first 150 lines
    if num < 150:
        for s in remove_startswith:
            if sent.startswith(s):
                return ''
        for r in remove_regex:
            if r.match(sent) is not None:
                return ''

    if count(sent, '"') % 2 == 1:
        sent = sent.replace('"', '')
    if count(sent, '“') != count(sent, '”'):
        sent = sent.replace('“', '')
        sent = sent.replace('”', '')
    if count(sent, '(') != count(sent, ')'):
        sent = sent.replace('(', '')
        sent = sent.replace(')', '')

    return sent


def main(args):
    nltk.download('punkt')
    with open(args.output, 'w') as out:
        for root, _, files in os.walk(args.dir):
            for file_name in files:
                wrote_sents = 0
                with open(os.path.join(root, file_name), 'r', encoding='utf8', errors='ignore') as fin:
                    for i, line in enumerate(fin):
                        line = line.strip()
                        sents = nltk.sent_tokenize(line)
                        for sent in sents:
                            sent = normalize(sent, i)
                            if len(sent) == 0:
                                continue
                            print(sent, file=out)
                            wrote_sents += 1
                if wrote_sents > 0:
                    print('', file=out)
                    print(f'Wrote {wrote_sents} sentences from {file_name}')
                else:
                    print(f'!!! NO SENTENCES FOUND IN {file_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dir',
        metavar='DIR',
        help='directory containing documents to process.',
    )
    parser.add_argument(
        '--output',
        required=True,
        metavar='FILE',
    )
    args = parser.parse_args()
    print(args)
    main(args)
