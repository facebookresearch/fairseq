#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import argparse

from fairseq.data import dictionary
from fairseq.data import IndexedDataset


def get_parser():
    parser = argparse.ArgumentParser(
        description='writes text from binarized file to stdout')
    # fmt: off
    parser.add_argument('--dict', metavar='FP', required=True, help='dictionary containing known words')
    parser.add_argument('--input', metavar='FP', required=True, help='binarized file to read')
    # fmt: on

    return parser


def main(args):
    dict = dictionary.Dictionary.load(args.dict)
    ds = IndexedDataset(args.input, fix_lua_indexing=True)
    for tensor_line in ds:
        print(dict.string(tensor_line))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
