# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



import os
import glob
import argparse
from utils.dedup import deup

import sys
WORKDIR_ROOT = os.environ.get('WORKDIR_ROOT', None)

if WORKDIR_ROOT is None or  not WORKDIR_ROOT.strip():
    print('please specify your working directory root in OS environment variable WORKDIR_ROOT. Exitting..."')
    sys.exit(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-folder", type=str, required=True,
                        help="the data folder to be dedup")
    parser.add_argument("--to-folder", type=str, required=True,
                        help="the data folder to save deduped data")
    parser.add_argument('--directions', type=str, default=None, required=False)

    args = parser.parse_args()    

    if args.directions is None:
        raw_files = glob.glob(f'{args.from_folder}/train*')

        directions = [os.path.split(file_path)[-1].split('.')[1] for file_path in raw_files]
    else:
        directions = args.directions.split(',')
    directions = sorted(set(directions))
    
    for direction in directions:
        src, tgt = direction.split('-')
        src_file = f'{args.from_folder}/train.{src}-{tgt}.{src}'
        tgt_file = f'{args.from_folder}/train.{src}-{tgt}.{tgt}'
        src_file_out = f'{args.to_folder}/train.{src}-{tgt}.{src}'
        tgt_file_out = f'{args.to_folder}/train.{src}-{tgt}.{tgt}'
        assert src_file != src_file_out
        assert tgt_file != tgt_file_out
        print(f'deduping {src_file}, {tgt_file}')
        deup(src_file, tgt_file, src_file_out, tgt_file_out)
                

if __name__ == "__main__":
    main()
