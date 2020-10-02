#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sentencepiece as spm
import logging
log = logging.getLogger(__name__)

# tsv columns
# id	audio	n_frames	tgt_text	speaker	src_text	src_lang	tgt_lang
def load_values(file_name, key, value_table=None, spm_name="", utt_id_list=None):

    if spm_name != "":
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_name)
    else:
        sp = None
    value_table = {} if value_table is None else value_table
    with open(file_name, 'r') as f:
        line_index = 0
        for line in f.readlines():
            line = line.strip()
            try:
                (utt_id, value) = line.split("\t", 1)
            except ValueError:
                value = line
                utt_id = utt_id_list[line_index]
                line_index += 1
            if utt_id not in value_table:
                value_table[utt_id] = {}
            if sp is not None:
                #import pdb;pdb.set_trace()
                try:
                    value = " ".join(sp.EncodeAsPieces(value))
                except:
                    import pdb;pdb.set_trace()
            value_table[utt_id][key]=value
    return value_table

def expand_audio(value_table):
    for uid in value_table:
        audio_path, n_frames = value_table[uid]['audio'].split("\t", 1)
        value_table[uid]['audio']=audio_path
        value_table[uid]['n_frames']=n_frames
    return value_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-tsv", required=True,
                        help="aggregated audio files")
    parser.add_argument("--tgt-labels", required=True,
                        help="aggregated target labels ")
    parser.add_argument("--src-labels", default="", 
                        help="aggregated source labels ")
    parser.add_argument("--output-tsv", required=True,
                        help="path to save tsv output")
    parser.add_argument("--tgt-spm", default="",
                        help="sentencepiece model to use for target encoding"),
    args = parser.parse_args()

    cols = ['id', 'audio', 'n_frames', 'tgt_text']
    value_table = load_values(args.audio_tsv, "audio")
    value_table = expand_audio(value_table)
    value_table = load_values(args.tgt_labels, "tgt_text", value_table, args.tgt_spm, list(value_table.keys())) 
    if args.src_labels != "":
        value_table = load_values(args.src_labels, "src_text", value_table) 
        cols = cols.append('src_text')
    with open(args.output_tsv,'w') as f:
        f.write("\t".join(cols) + "\n")
        outlines=[]
        for utt_id in value_table.keys():
            outline = [utt_id]
            for col in cols[1:]:
                if col not in value_table[utt_id]:
                    logging.warning(f"{utt_id} doesn't has {col}")
                    outline = None 
                    break
                outline.append(value_table[utt_id][col])
            if outline is not None:
                outlines.append(outline) 
        f.write("\n".join(["\t".join(ol) for ol in outlines]) + "\n")
        

if __name__ == "__main__":
    main()
