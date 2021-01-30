# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse
import pandas as pd
import sys


WORKDIR_ROOT = os.environ.get('WORKDIR_ROOT', None)

if WORKDIR_ROOT is None or  not WORKDIR_ROOT.strip():
    print('please specify your working directory root in OS environment variable WORKDIR_ROOT. Exitting..."')
    sys.exit(-1)

def load_langs(path):
    with open(path) as fr:
        langs = [l.strip() for l in fr]
    return langs



def load_sentences(raw_data, split, direction):
    src, tgt = direction.split('-')
    src_path = f"{raw_data}/{split}.{direction}.{src}"
    tgt_path = f"{raw_data}/{split}.{direction}.{tgt}"
    if os.path.exists(src_path) and os.path.exists(tgt_path):
        return [(src, open(src_path).read().splitlines()), (tgt, open(tgt_path).read().splitlines())]
    else:
        return []

def swap_direction(d):
    src, tgt = d.split('-')
    return f'{tgt}-{src}'

def get_all_test_data(raw_data, directions, split='test'):
    test_data = [ 
        x
        for dd in directions
        for d in [dd, swap_direction(dd)]
        for x in load_sentences(raw_data, split, d)
    ]
    # all_test_data = {s for _, d in test_data for s in d}
    all_test_data = {}
    for lang, d in test_data:
        for s in d:
            s = s.strip()
            lgs = all_test_data.get(s, set())
            lgs.add(lang)
            all_test_data[s] = lgs
    return all_test_data, test_data


def check_train_sentences(src_path, tgt_path, direction, all_test_data, mess_up_train={}):
    # src, tgt = direction.split('-')
    print(f'check training data for {direction} in {src_path} and {tgt_path}')
    size = 0
    overlapped_size_counted_dup = 0
    if not os.path.exists(tgt_path) or not os.path.exists(src_path):
        return mess_up_train, size, overlapped_size_counted_dup

    with open(src_path) as f, open(tgt_path) as g:
        for src_line, tgt_line in zip(f, g):
            s = src_line.strip()
            t = tgt_line.strip()
            size += 1
            if  s in all_test_data:
                langs = mess_up_train.get(s, set())
                langs.add(direction)
                mess_up_train[s] = langs
                overlapped_size_counted_dup += 1
            if t in all_test_data:
                langs = mess_up_train.get(t, set())
                langs.add(direction)
                mess_up_train[t] = langs 
                overlapped_size_counted_dup += 1
    print(f'{direction}: size={size}, overlapped={overlapped_size_counted_dup}')
    return mess_up_train, size, overlapped_size_counted_dup

def check_train_all(raw_data, directions, all_test_data):
    mess_up_train = {}
    data_sizes = {}
    # raw_data = '~chau/data-bin/MineBART/multilingual_mined_100M/en_XX/et_EE-en_XX/all.{en_XX, et_EE}'
    print(f'checking training data againsts # {len(all_test_data)} sentences')
    print(f'example test data: ', [s for i, s in enumerate(all_test_data.keys()) if i < 10])
    for direction in directions:
        src, tgt = direction.split('-')
        path = f'{raw_data}/en_XX/{direction}/all'
        src_path = f'{path}.{src}'
        tgt_path = f'{path}.{tgt}'
        print(f'checking {src_path} {tgt_path}')
        _, size, overlapped_size_counted_dup = check_train_sentences(src_path, tgt_path, direction, all_test_data, mess_up_train)
        data_sizes[direction] = (size, overlapped_size_counted_dup)
    return mess_up_train, data_sizes




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True,
                        help="the data folder ")
    parser.add_argument("--test-data", type=str, required=True,
                        help="the test data folder ")                        
    parser.add_argument('--directions', type=str, default=None, required=False)

    args = parser.parse_args()    
    directions = args.directions.split(',')
    directions = sorted(set(directions))

    results = []
    # print(f'checking where {args.split} split data are in training')
    # print(f'direction\tcommon_count\tsrc common\ttgt common\tfrom_size\tto_size')
    raw_data = args.folder
    all_test_data, test_data = get_all_test_data(args.test_data, directions, split='test')
    mess_up_train, data_sizes = check_train_all(raw_data, directions, all_test_data)
    print(data_sizes)


if __name__ == "__main__":
    main()
