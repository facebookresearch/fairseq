# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse

def deup(src_file, tgt_file, src_file_out, tgt_file_out):
    seen = set()
    dup_count = 0
    with open(src_file, encoding='utf-8') as fsrc, \
        open(tgt_file, encoding='utf-8') as ftgt, \
        open(src_file_out, 'w', encoding='utf-8') as fsrc_out, \
        open(tgt_file_out, 'w', encoding='utf-8') as ftgt_out:
        for s, t in zip(fsrc, ftgt):
            if (s, t) not in seen:
                fsrc_out.write(s)
                ftgt_out.write(t)   
                seen.add((s, t))
            else:
                dup_count += 1
    print(f'number of duplication: {dup_count}')    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file", type=str, required=True,
                        help="src file")
    parser.add_argument("--tgt-file", type=str, required=True,
                        help="tgt file")
    parser.add_argument("--src-file-out", type=str, required=True,
                        help="src ouptut file")
    parser.add_argument("--tgt-file-out", type=str, required=True,
                        help="tgt ouput file") 
    args = parser.parse_args()    
    deup(args.src_file, args.tgt_file, args.src_file_out, args.tgt_file_out)
                

if __name__ == "__main__":
    main()
