#!/usr/bin/env python3

import argparse
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--inputs',
        required=True,
        nargs='+',
        help='files to process.',
    )
    parser.add_argument(
        '--output',
        required=True,
        metavar='FILE',
        help='Path for output',
    )

    parser.add_argument(
        '--separator',
        default='\t',
        metavar='SEP',
        help='separator between columns',
    )

    args = parser.parse_args()
    print(args)

    labels = {
        'neutral': 0,
        'entailment': 1,
        'contradiction': 2,
    }

    for inp in args.inputs:
        filename = os.path.basename(inp)
        base_filename = os.path.splitext(filename)[0]
        s1_filename = base_filename + '1.txt'
        s2_filename = base_filename + '2.txt'
        label_filename = base_filename + '.lbl'

        sent1_col = sent2_col = label_col = None

        with open(inp, 'r') as f_in, open(os.path.join(args.output, s1_filename), 'w') as s1_out, open(
                os.path.join(args.output, s2_filename), 'w') as s2_out, open(
                os.path.join(args.output, label_filename), 'w') as lbl_out:
            for line in f_in:
                parts = line.strip().split(args.separator)

                if sent1_col is None:
                    sent1_col = parts.index('sentence1')
                    sent2_col = parts.index('sentence2')
                    label_col = parts.index('gold_label')
                    continue


                if parts[label_col] == '-':
                    continue

                if parts[label_col] == '' or parts[label_col] == '-':
                    print(inp)
                    print(parts)
                    print(line)
                    print(sent1_col, sent2_col, label_col)
                    print(parts[sent1_col])
                    print(parts[sent2_col])
                label = labels[parts[label_col]]
                print(label, file=lbl_out)
                print(parts[sent1_col], file=s1_out)
                print(parts[sent2_col], file=s2_out)


if __name__ == '__main__':
    main()
