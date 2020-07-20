#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fileinput
import sacremoses


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('files', nargs='*', help='input files')
    args = parser.parse_args()

    detok = sacremoses.MosesDetokenizer()

    for line in fileinput.input(args.files, openhook=fileinput.hook_compressed):
        print(detok.detokenize(line.strip().split(' ')).replace(' @', '').replace('@ ', '').replace(' =', '=').replace('= ', '=').replace(' – ', '–'))


if __name__ == '__main__':
    main()
