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
        '--label-col',
        default=1,
        metavar='N',
        help='column for the label (0-based indexing)',
    )

    parser.add_argument(
        '--data-col',
        type=int,
        default=3,
        metavar='N',
        help='column for the data (0-based indexing)',
    )

    parser.add_argument(
        '--separator',
        default='\t',
        metavar='SEP',
        help='separator between columns',
    )

    args = parser.parse_args()
    print(args)

    for inp in args.inputs:
        filename = os.path.basename(inp)
        base_filename = os.path.splitext(filename)[0]
        data_filename = base_filename + '.txt'
        label_filename = base_filename + '.lbl'
        with open(inp, 'r') as f_in, open(os.path.join(args.output, data_filename), 'w') as data_out, open(
                os.path.join(args.output, label_filename), 'w') as lbl_out:
            for line in f_in:
                parts = line.strip().split(args.separator)
                print(parts[args.label_col], file=lbl_out)
                print(parts[args.data_col], file=data_out)


if __name__ == '__main__':
    main()
