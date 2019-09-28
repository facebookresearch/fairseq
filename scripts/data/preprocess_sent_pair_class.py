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
        metavar='FILE',
        help='Path for output',
    )

    parser.add_argument(
        '--label-col-name',
        default='gold_label',
        metavar='NAME',
        help='the name of the label column',
    )

    parser.add_argument(
        '--sentence1-col-name',
        default='sentence1',
        metavar='NAME',
        help='the name of the column holding sentence 1',
    )

    parser.add_argument(
        '--sentence2-col-name',
        default='sentence2',
        metavar='NAME',
        help='the name of the column holding sentence 2',
    )

    parser.add_argument(
        '--labels',
        default=['neutral', 'entailment', 'contradiction'],
        nargs='+',
        help='list of labels',
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

    if len(args.labels) == 1 and args.labels[0] == 'identity':
        labels = None
    else:
        labels = {l: i for i, l in enumerate(args.labels)}
        print (labels)
    for inp in args.inputs:
        filename = os.path.basename(inp)
        base_filename = os.path.splitext(filename)[0]
        s1_filename = base_filename + '1.txt'
        s2_filename = base_filename + '2.txt'
        label_filename = base_filename + '.lbl'

        sent1_col = sent2_col = label_col = None

        with open(inp, 'r', encoding='utf-8-sig') as f_in, open(os.path.join(args.output, s1_filename),
                                                                'w') as s1_out, open(
                os.path.join(args.output, s2_filename), 'w') as s2_out, open(
            os.path.join(args.output, label_filename), 'w') as lbl_out:
            for line in f_in:
                parts = line.strip().split(args.separator)

                if sent1_col is None:
                    sent1_col = parts.index(args.sentence1_col_name)
                    sent2_col = parts.index(args.sentence2_col_name)
                    label_col = parts.index(args.label_col_name)
                    continue

                if len(parts) <= label_col:
                    print('invalid line', parts)
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
                label = labels[parts[label_col]] if labels is not None else parts[label_col]
                print(label, file=lbl_out)
                txt1 = parts[sent1_col]
                tokenized_text1 = tokenizer.tokenize(txt1)
                txt2 = parts[sent2_col]
                tokenized_text2 = tokenizer.tokenize(txt2)
                print(" ".join(tokenized_text1), file=s1_out)
                print(" ".join(tokenized_text2), file=s2_out)


if __name__ == '__main__':
    main()
