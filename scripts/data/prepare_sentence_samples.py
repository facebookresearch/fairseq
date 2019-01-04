#!/usr/bin/env python3

import argparse
import random


def main(args):
    docs = []
    with open(args.input, 'r') as inp:
        doc = []
        for line in inp:
            line = line.strip().split()
            if len(line) == 0 and len(doc) > 0:
                docs.append(doc)
                doc = []
            elif len(line) > 0:
                doc.append(line)

    def get_seq_len():
        if args.short_seq_prob == 0:
            return args.seq_length
        else:
            return args.seq_length if random.random() > args.short_seq_prob else random.randint(2, args.seq_length)

    with open(args.output, 'w') as out, open(args.output + '.lbl', 'w') as lbl:
        assert len(docs) > 1
        for _ in range(args.dupe_factor):
            for i, doc in enumerate(docs):
                j = 0
                chunk = []
                length = 0
                seq_len = get_seq_len()
                while j < len(doc):
                    chunk.append(doc[j])
                    length += len(doc[j])
                    if j == len(doc) - 1 or length >= seq_len:
                        a_end = 1 if len(chunk) == 1 else random.randint(1, len(chunk) - 1)
                        a_toks = [t for c in chunk[:a_end] for t in c]
                        b_toks = []
                        if args.randomize_next_sentence and (random.random() >= 0.5 or len(chunk) == 1):
                            r = i
                            while r == i:
                                r = random.randint(0, len(docs) - 1)
                            k = random.randint(0, len(docs[r]) - 1)
                            while k < len(docs[r]):
                                b_toks.extend(docs[r][k])
                                if (len(a_toks) + len(b_toks)) >= seq_len:
                                    break
                                k += 1
                            j -= len(chunk) - a_end
                            print(0, file=lbl)
                        else:
                            b_toks.extend(t for c in chunk[a_end:] for t in c)
                            if args.randomize_next_sentence:
                                print(1, file=lbl)
                        while (len(a_toks) + len(b_toks)) > seq_len:
                            target = a_toks if len(a_toks) > len(b_toks) else b_toks
                            target.pop(0 if random.random() >= 0.5 else -1)

                        assert len(a_toks) > 0
                        assert not args.randomize_next_sentence or len(b_toks) > 0

                        if len(b_toks) == 0:
                            sample = a_toks
                        else:
                            sample = a_toks + [args.sep] + b_toks
                        print(' '.join(sample), file=out)
                        chunk = []
                        length = 0
                        seq_len = get_seq_len()
                    j += 1


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
    # parser.add_argument(
    #     '--keep-single',
    #     action='store_true',
    #     help='if set, keeps single example sentences also (e.g. for uneven length docs)'
    # )
    parser.add_argument(
        '--randomize-next-sentence',
        action='store_true',
        help='if set, next sentence has a 50% chance of being from a different document, and the results are stored in .lbl file'
    )
    parser.add_argument(
        '--seq-length',
        default=512,
        type=int,
        help='if set, sequences are constructed by taking up to this many tokens. when the number of tokens exceeds this length, they are truncated. sentences are assigned to first or second segment randomly'
    )
    parser.add_argument(
        '--dupe-factor',
        default=1,
        type=int,
        help='how many times to generate samples from each document (affects where the examples are cut)'
    )
    parser.add_argument(
        '--short-seq-prob',
        default=0,
        type=float,
        help='the likelihood of generating a sequence that is shorter than seq-length'
    )
    args = parser.parse_args()
    print(args)
    main(args)
