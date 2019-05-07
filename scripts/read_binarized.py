#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse

from fairseq.data import Dictionary
from fairseq.data import indexed_dataset


def get_parser():
    parser = argparse.ArgumentParser(
        description='writes text from binarized file to stdout')
    # fmt: off
    parser.add_argument('--dataset-impl', help='dataset implementation',
                        choices=['raw', 'lazy', 'cached', 'mmap'], default='lazy')
    parser.add_argument('--dict', metavar='FP', help='dictionary containing known words', default=None)
    parser.add_argument('--input', metavar='FP', required=True, help='binarized file to read')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    dictionary = Dictionary.load(args.dict) if args.dict is not None else None
    dataset = indexed_dataset.make_dataset(args.input, impl=args.dataset_impl,
                                           fix_lua_indexing=True, dictionary=dictionary)

    for tensor_line in dataset:
        if dictionary is None:
            line = ' '.join([str(int(x)) for x in tensor_line])
        else:
            line = dictionary.string(tensor_line)

        print(line)


if __name__ == '__main__':
    main()
