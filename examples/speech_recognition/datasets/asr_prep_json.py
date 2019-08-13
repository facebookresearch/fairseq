#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple
import concurrent.futures
from itertools import chain
import argparse
import os
import json
import multiprocessing
import torchaudio

from fairseq.data import Dictionary

MILLISECONDS_TO_SECONDS = 0.001


def process_samples(samples, labels, sp_path, tgt_dict):
    utts = {}

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)

    for s in samples:
        lable = labels[s.utt_id]
        input = {}
        output = {}
        si, ei = torchaudio.info(s.aud_path)
        input["length_ms"] = int(
            si.length / si.channels / si.rate / MILLISECONDS_TO_SECONDS)
        input["path"] = s.aud_path

        token = " ".join(sp.EncodeAsPieces(lable))
        ids = tgt_dict.encode_line(token, append_eos=False)
        output["text"] = lable
        output["token"] = token
        output["tokenid"] = ', '.join(map(str, [t.tolist() for t in ids]))
        utts[s.utt_id] = {"input": input, "output": output}
    return utts


Sample = namedtuple('Sample', 'aud_path utt_id')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dirs", nargs="+", default=['-'], required=True,
                        help="input directories with audio files")
    parser.add_argument("--labels", required=True,
                        help="aggregated input labels with format <ID LABEL> per line",
                        type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument("--spm-model", required=True,
                        help="sentencepiece model to use for encoding")
    parser.add_argument("--dictionary", required=True,
                        help="file to load fairseq dictionary from",
                        type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument(
        "--audio-format", choices=["flac", "wav"], default="wav")
    parser.add_argument("--output", required=True, type=argparse.FileType('w'),
                        help="path to save json output")
    parser.add_argument("--num-cpu", default=1, type=int, required=True)
    args = parser.parse_args()

    tgt_dict = Dictionary.load(args.dictionary)

    labels = {}
    for line in args.labels:
        (utt_id, label) = line.split(" ", 1)
        labels[utt_id] = label
    if len(labels) == 0:
        raise Exception('No labels found in ', args.labels_path)

    samples = []
    for path, _, files in chain.from_iterable(os.walk(path) for path in args.audio_dirs):
        for f in files:
            if f.endswith(args.audio_format):
                if len(os.path.splitext(f)) != 2:
                    raise Exception(
                        'Expect <utt_id.extension> file name. Got: ', f)
                utt_id = os.path.splitext(f)[0]
                if utt_id not in labels:
                    continue
                samples.append(Sample(os.path.join(path, f), utt_id))

    utts = {}
    num_cpu = args.num_cpu
    chunk_size = int(1 + (len(samples) / num_cpu))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
        samples_chunks = [samples[i:i + chunk_size]
                          for i in range(0, len(samples), chunk_size)]
        futures = [executor.submit(
            process_samples, chunk, labels, args.spm_model, tgt_dict) for chunk in samples_chunks]
        for future in futures:
            try:
                data = future.result()
            except Exception as exc:
                print('generated an exception: ', exc)
            else:
                utts.update(data)
    json.dump({"utts": utts}, args.output, indent=4)


if __name__ == "__main__":
    main()
