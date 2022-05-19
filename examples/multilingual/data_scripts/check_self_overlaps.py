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

def get_directions(folder):
    raw_files = glob.glob(f'{folder}/train*')
    directions = [os.path.split(file_path)[-1].split('.')[1] for file_path in raw_files] 
    return directions   

def diff_list(lhs, rhs):
    return set(lhs).difference(set(rhs))

def check_diff(
    from_src_file, from_tgt_file, 
    to_src_file, to_tgt_file, 
):
    seen_in_from = set()
    seen_src_in_from = set()
    seen_tgt_in_from = set()
    from_count = 0
    with open(from_src_file, encoding='utf-8') as fsrc, \
        open(from_tgt_file, encoding='utf-8') as ftgt:
        for s, t in zip(fsrc, ftgt):
            seen_in_from.add((s, t))
            seen_src_in_from.add(s)
            seen_tgt_in_from.add(t)
            from_count += 1
    common = 0
    common_src = 0
    common_tgt = 0
    to_count = 0
    seen = set()

    with open(to_src_file, encoding='utf-8') as fsrc, \
        open(to_tgt_file, encoding='utf-8') as ftgt:
        for s, t in zip(fsrc, ftgt):
            to_count += 1
            if (s, t) not in seen:
                if (s, t) in seen_in_from:
                    common += 1
                if s in seen_src_in_from:
                    common_src += 1
                    seen_src_in_from.remove(s)
                if t in seen_tgt_in_from:
                    common_tgt += 1
                    seen_tgt_in_from.remove(t)
                seen.add((s, t))
    return common, common_src, common_tgt, from_count, to_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True,
                        help="the data folder ")
    parser.add_argument("--split", type=str, default='test',
                        help="split (valid, test) to check against training data")
    parser.add_argument('--directions', type=str, default=None, required=False)

    args = parser.parse_args()    

    if args.directions is None:
        directions = set(get_directions(args.folder))
        directions = sorted(directions)
    else:
        directions = args.directions.split(',')
    directions = sorted(set(directions))

    results = []
    print(f'checking where {args.split} split data are in training')
    print(f'direction\tcommon_count\tsrc common\ttgt common\tfrom_size\tto_size')

    for direction in directions:
        src, tgt = direction.split('-')
        from_src_file = f'{args.folder}/{args.split}.{src}-{tgt}.{src}'
        from_tgt_file = f'{args.folder}/{args.split}.{src}-{tgt}.{tgt}'
        if not os.path.exists(from_src_file):
            # some test/valid data might in reverse directinos:
            from_src_file = f'{args.folder}/{args.split}.{tgt}-{src}.{src}'
            from_tgt_file = f'{args.folder}/{args.split}.{tgt}-{src}.{tgt}'            
        to_src_file = f'{args.folder}/train.{src}-{tgt}.{src}'
        to_tgt_file = f'{args.folder}/train.{src}-{tgt}.{tgt}'
        if not os.path.exists(to_src_file) or not os.path.exists(from_src_file):
            continue
        r = check_diff(from_src_file, from_tgt_file, to_src_file, to_tgt_file)
        results.append(r)
        print(f'{direction}\t', '\t'.join(map(str, r)))
                

if __name__ == "__main__":
    main()
