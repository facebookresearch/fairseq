#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append('/private/home/yinhanliu/pytorch-pretrained-BERT')
from pytorch_pretrained_bert.tokenization import BertTokenizer
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
        metavar='DIR',
        help='Path for output',
    )

    parser.add_argument(
        '--label-col',
        type=int,
        default=1,
        metavar='N',
        help='column for the label (0-based indexing)',
    )

    parser.add_argument(
        '--skip-rows',
        type=int,
        default=0,
        metavar='N',
        help='number of rows to skip',
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for inp in args.inputs:
        filename = os.path.basename(inp)
        base_filename = os.path.splitext(filename)[0]
        data_filename = base_filename + '.txt'
        label_filename = base_filename + '.lbl'
        with open(inp, 'r') as f_in, open(os.path.join(args.output, data_filename), 'w') as data_out, open(
                os.path.join(args.output, label_filename), 'w') as lbl_out:
            for i, line in enumerate(f_in):
                if i < args.skip_rows:
                    continue
                parts = line.strip().split(args.separator)
                if args.label_col >= 0:
                    print(parts[args.label_col], file=lbl_out)
                txt = parts[args.data_col]
                tokenized_text = tokenizer.tokenize(txt)
                print(" ".join(tokenized_text), file=data_out)
        if args.label_col < 0:
            os.remove(os.path.join(args.output, label_filename))


if __name__ == '__main__':
    main()
