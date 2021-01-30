#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from collections import namedtuple
from itertools import chain

import sentencepiece as spm
from fairseq.data import Dictionary


MILLISECONDS_TO_SECONDS = 0.001


def process_sample(aud_path, lable, utt_id, sp, tgt_dict):
    import torchaudio

    input = {}
    output = {}
    si, ei = torchaudio.info(aud_path)
    input["length_ms"] = int(
        si.length / si.channels / si.rate / MILLISECONDS_TO_SECONDS
    )
    input["path"] = aud_path

    token = " ".join(sp.EncodeAsPieces(lable))
    ids = tgt_dict.encode_line(token, append_eos=False)
    output["text"] = lable
    output["token"] = token
    output["tokenid"] = ", ".join(map(str, [t.tolist() for t in ids]))
    return {utt_id: {"input": input, "output": output}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-dirs",
        nargs="+",
        default=["-"],
        required=True,
        help="input directories with audio files",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="aggregated input labels with format <ID LABEL> per line",
        type=argparse.FileType("r", encoding="UTF-8"),
    )
    parser.add_argument(
        "--spm-model",
        required=True,
        help="sentencepiece model to use for encoding",
        type=argparse.FileType("r", encoding="UTF-8"),
    )
    parser.add_argument(
        "--dictionary",
        required=True,
        help="file to load fairseq dictionary from",
        type=argparse.FileType("r", encoding="UTF-8"),
    )
    parser.add_argument("--audio-format", choices=["flac", "wav"], default="wav")
    parser.add_argument(
        "--output",
        required=True,
        type=argparse.FileType("w"),
        help="path to save json output",
    )
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model.name)

    tgt_dict = Dictionary.load(args.dictionary)

    labels = {}
    for line in args.labels:
        (utt_id, label) = line.split(" ", 1)
        labels[utt_id] = label
    if len(labels) == 0:
        raise Exception("No labels found in ", args.labels_path)

    Sample = namedtuple("Sample", "aud_path utt_id")
    samples = []
    for path, _, files in chain.from_iterable(
        os.walk(path) for path in args.audio_dirs
    ):
        for f in files:
            if f.endswith(args.audio_format):
                if len(os.path.splitext(f)) != 2:
                    raise Exception("Expect <utt_id.extension> file name. Got: ", f)
                utt_id = os.path.splitext(f)[0]
                if utt_id not in labels:
                    continue
                samples.append(Sample(os.path.join(path, f), utt_id))

    utts = {}
    num_cpu = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpu) as executor:
        future_to_sample = {
            executor.submit(
                process_sample, s.aud_path, labels[s.utt_id], s.utt_id, sp, tgt_dict
            ): s
            for s in samples
        }
        for future in concurrent.futures.as_completed(future_to_sample):
            try:
                data = future.result()
            except Exception as exc:
                print("generated an exception: ", exc)
            else:
                utts.update(data)
    json.dump({"utts": utts}, args.output, indent=4)


if __name__ == "__main__":
    main()
