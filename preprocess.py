#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
from collections import Counter
from itertools import zip_longest
import os
import shutil


from fairseq.data import indexed_dataset, dictionary
from fairseq.tokenizer import Tokenizer, tokenize_line
from multiprocessing import Pool, Manager, Process


def get_parser():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-s", "--source-lang", default=None, metavar="SRC",
                        help="source language")
    parser.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                        help="target language")
    parser.add_argument("--trainpref", metavar="FP", default=None,
                        help="train file prefix")
    parser.add_argument("--validpref", metavar="FP", default=None,
                        help="comma separated, valid file prefixes")
    parser.add_argument("--testpref", metavar="FP", default=None,
                        help="comma separated, test file prefixes")
    parser.add_argument("--destdir", metavar="DIR", default="data-bin",
                        help="destination dir")
    parser.add_argument("--thresholdtgt", metavar="N", default=0, type=int,
                        help="map words appearing less than threshold times to unknown")
    parser.add_argument("--thresholdsrc", metavar="N", default=0, type=int,
                        help="map words appearing less than threshold times to unknown")
    parser.add_argument("--tgtdict", metavar="FP",
                        help="reuse given target dictionary")
    parser.add_argument("--srcdict", metavar="FP",
                        help="reuse given source dictionary")
    parser.add_argument("--nwordstgt", metavar="N", default=-1, type=int,
                        help="number of target words to retain")
    parser.add_argument("--nwordssrc", metavar="N", default=-1, type=int,
                        help="number of source words to retain")
    parser.add_argument("--alignfile", metavar="ALIGN", default=None,
                        help="an alignment file (optional)")
    parser.add_argument("--output-format", metavar="FORMAT", default="binary",
                        choices=["binary", "raw"],
                        help="output format (optional)")
    parser.add_argument("--joined-dictionary", action="store_true",
                        help="Generate joined dictionary")
    parser.add_argument("--only-source", action="store_true",
                        help="Only process the source language")
    parser.add_argument("--padding-factor", metavar="N", default=8, type=int,
                        help="Pad dictionary size to be multiple of N")
    parser.add_argument("--workers", metavar="N", default=1, type=int,
                        help="number of parallel workers")
    # fmt: on
    return parser


def main(args):
    print(args)
    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    if args.joined_dictionary:
        assert not args.srcdict, "cannot combine --srcdict and --joined-dictionary"
        assert not args.tgtdict, "cannot combine --tgtdict and --joined-dictionary"
        src_dict = build_dictionary(
            {train_path(lang) for lang in [args.source_lang, args.target_lang]},
            args.workers,
        )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = dictionary.Dictionary.load(args.srcdict)
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], args.workers)
        if target:
            if args.tgtdict:
                tgt_dict = dictionary.Dictionary.load(args.tgtdict)
            else:
                assert (
                    args.trainpref
                ), "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary(
                    [train_path(args.target_lang)], args.workers
                )

    src_dict.finalize(
        threshold=args.thresholdsrc,
        nwords=args.nwordssrc,
        padding_factor=args.padding_factor,
    )
    src_dict.save(dict_path(args.source_lang))
    if target:
        if not args.joined_dictionary:
            tgt_dict.finalize(
                threshold=args.thresholdtgt,
                nwords=args.nwordstgt,
                padding_factor=args.padding_factor,
            )
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(input_prefix, output_prefix, lang, num_workers):
        dict = dictionary.Dictionary.load(dict_path(lang))
        print("| [{}] Dictionary: {} types".format(lang, len(dict) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Tokenizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        dict,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.IndexedDatasetBuilder(
            dataset_dest_file(args, output_prefix, lang, "bin")
        )
        merge_result(
            Tokenizer.binarize(
                input_file, dict, lambda t: ds.add_item(t), offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                dict.unk_word,
            )
        )

    def make_dataset(input_prefix, output_prefix, lang, num_workers=1):
        if args.output_format == "binary":
            make_binary_dataset(input_prefix, output_prefix, lang, num_workers)
        elif args.output_format == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)

    def make_all(lang):
        if args.trainpref:
            make_dataset(args.trainpref, "train", lang, num_workers=args.workers)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(validpref, outprefix, lang)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(testpref, outprefix, lang)

    make_all(args.source_lang)
    if target:
        make_all(args.target_lang)

    print("| Wrote preprocessed data to {}".format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        src_dict = dictionary.Dictionary.load(dict_path(args.source_lang))
        tgt_dict = dictionary.Dictionary.load(dict_path(args.target_lang))
        freq_map = {}
        with open(args.alignfile, "r") as align_file:
            with open(src_file_name, "r") as src_file:
                with open(tgt_file_name, "r") as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = Tokenizer.tokenize(s, src_dict, add_if_not_exist=False)
                        ti = Tokenizer.tokenize(t, tgt_dict, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
            os.path.join(
                args.destdir,
                "alignment.{}-{}.txt".format(args.source_lang, args.target_lang),
            ),
            "w",
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def build_and_save_dictionary(
    train_path, output_path, num_workers, freq_threshold, max_words
):
    dict = build_dictionary([train_path], num_workers)
    dict.finalize(threshold=freq_threshold, nwords=max_words)
    dict_path = os.path.join(output_path, "dict.txt")
    dict.save(dict_path)
    return dict_path


def build_dictionary(filenames, workers):
    d = dictionary.Dictionary()
    for filename in filenames:
        Tokenizer.add_file_to_dictionary(filename, d, tokenize_line, workers)
    return d


def binarize(args, filename, dict, output_prefix, lang, offset, end):
    ds = indexed_dataset.IndexedDatasetBuilder(
        dataset_dest_file(args, output_prefix, lang, "bin")
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Tokenizer.binarize(filename, dict, consumer, offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_with_load(args, filename, dict_path, output_prefix, lang, offset, end):
    dict = dictionary.Dictionary.load(dict_path)
    binarize(args, filename, dict, output_prefix, lang, offset, end)
    return dataset_dest_prefix(args, output_prefix, lang)


def dataset_dest_prefix(args, output_prefix, lang):
    base = f"{args.destdir}/{output_prefix}"
    lang_part = (
        f".{args.source_lang}-{args.target_lang}.{lang}" if lang is not None else ""
    )
    return f"{base}{lang_part}"


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return f"{base}.{extension}"


def get_offsets(input_file, num_workers):
    return Tokenizer.find_offsets(input_file, num_workers)


def merge_files(files, outpath):
    ds = indexed_dataset.IndexedDatasetBuilder("{}.bin".format(outpath))
    for file in files:
        ds.merge_file_(file)
        os.remove(indexed_dataset.data_file_path(file))
        os.remove(indexed_dataset.index_file_path(file))
    ds.finalize("{}.idx".format(outpath))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
