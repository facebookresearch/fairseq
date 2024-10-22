# Copyright (c) Carnegie Mellon University (Jiatong Shi)
#
# # This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os


def split_to_letter(tgt, dictset):
   tgt_result = tgt.split(maxsplit=1)
   tgt_id=tgt_result[0]
   tgt_text=tgt_result[1]

   tgt_letters = [(char if char != " " else "<space>" ) for char in tgt_text]
   for letters in tgt_letters:
       if letters not in dictset:
           dictset[letters]=1
       else:
           dictset[letters] += 1
   return tgt_id, " ".join(tgt_letters), dictset


def main(args):
    src_dict = {}
    tgt_dict = {}
    for subset in args.subset:
        src_text = open("{}/text.{}".format(args.src_text, subset), "r", encoding="utf-8")
        tgt_text = open("{}/text.{}".format(args.tgt_text, subset), "r", encoding="utf-8")
        src_output = open("{}/source_letter/{}.tsv".format(args.dumpdir, subset), "w", encoding="utf-8")
        tgt_output = open("{}/target_letter/{}.tsv".format(args.dumpdir, subset), "w", encoding="utf-8")
        tgt = tgt_text.read().split("\n")
        src = src_text.read().split("\n")
        assert len(src) == len(tgt), "Mismatch source text and target text"

        src_output.write("id\ttgt_text\n")
        tgt_output.write("id\ttgt_text\n")
        for i in range(len(src)):
            if len(src[i].strip()) < 1:
                continue # skip empty line
            src_id, src_letters, src_dict = split_to_letter(src[i], src_dict)
            src_output.write("{}\t{}\n".format(src_id, src_letters))

            tgt_id, tgt_letters, tgt_dict = split_to_letter(tgt[i], tgt_dict)
            tgt_output.write("{}\t{}\n".format(tgt_id, tgt_letters))
    
    tgt_output_dict = open("{}/target_letter/dict.txt".format(args.dumpdir), "w", encoding="utf-8")
    src_output_dict = open("{}/source_letter/dict.txt".format(args.dumpdir), "w", encoding="utf-8")

    for l in tgt_dict:
        tgt_output_dict.write("{} {}\n".format(l, tgt_dict[l]))
    for l in src_dict:
        src_output_dict.write("{} {}\n".format(l, src_dict[l]))


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_text", type=str, required=True, help="source text folder"
    )
    parser.add_argument(
        "--tgt_text", type=str, required=True, help="target text folder"
    )
    parser.add_argument(
        "--subset", type=str, required=True, help="subset", action="append"
    )
    parser.add_argument(
        "--dumpdir", type=str, required=True, help="feature_dump_dir"
    )
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()