# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from collections import Counter
from dataclasses import dataclass
from glob import glob

import sentencepiece as spm
import xxhash
import yaml
from submitit import AutoExecutor, helpers

# MOSES special characters
REPLACEMENTS = {
    "&bar;": "|",
    "&#124;": "|",
    "&lt;": "<",
    "&gt;": ">",
    "&bra;": "[",
    "&ket;": "]",
    "&quot;": '"',
    "&apos;": "'",
    "&#91;": "[",
    "&#93;": "]",
    "&amp;": "&",
}


def moses_unescape(s):
    for before, after in REPLACEMENTS.items():
        s = s.replace(before, after)
    return s


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    empty_filtered: int = 0
    excessive_copy_filtered: int = 0
    duplicate_bt_filtered: int = 0


def try_decode_pair(bt_toks, orig_toks, spm_model, bt_lang, filter_stats, args):
    bt_toks = spm_model.encode(moses_unescape(bt_toks), out_type=str)
    orig_toks = moses_unescape(orig_toks).split(" ")
    bt_set = set(bt_toks)
    orig_set = set(orig_toks)
    # Basic rule-based filtering
    if not bt_toks or not orig_toks:
        filter_stats.empty_filtered += 1
        return None
    if len(bt_set & orig_set) / len(bt_set) > args.max_copy_ratio:
        filter_stats.excessive_copy_filtered += 1
        return None

    # Decode
    if spm_model is not None:
        orig = spm_model.decode(orig_toks)
        bt = spm_model.decode(bt_toks)
    else:
        orig = " ".join(orig_toks)

    return bt, orig


def process_lang(bt_lang, orig_lang, args):
    print(args)
    corpus_name = args.corpus_name
    # Get the list of shard outputs, ensuring that we only use one output per shard
    # in case there are multiple – prioritising the file that was modified last.
    src_pattern = os.path.join(
        args.base_folder, "src_sharded", orig_lang, f"{orig_lang}.*"
    )
    tgt_pattern = os.path.join(
        args.base_folder, "tgt_sharded", f"{orig_lang}-{bt_lang}", f"{bt_lang}.*"
    )
    src_shards = set(path[path.rindex(".") + 1 :] for path in glob(src_pattern))
    logging.info(f"Source shards: {src_shards}")
    tgt_shards = set(path[path.rindex(".") + 1 :] for path in glob(tgt_pattern))
    logging.info(f"Target shards: {src_shards}")
    in_common = list(src_shards & tgt_shards)
    logging.info(f"Common shards: {in_common}")

    orig_ins = [
        os.path.join(args.base_folder, "src_sharded", orig_lang, f"{orig_lang}.{n}")
        for n in in_common
    ]
    bt_ins = [
        os.path.join(
            args.base_folder, "tgt_sharded", f"{orig_lang}-{bt_lang}", f"{bt_lang}.{n}"
        )
        for n in in_common
    ]

    if args.spm_decode is not None:
        spm_model = spm.SentencePieceProcessor()
        spm_model.Load(args.spm_decode)
    else:
        spm_model = None

    corpus_root = os.path.join(args.output_folder, f"{bt_lang}-{orig_lang}")
    bt_tmp = os.path.join(corpus_root, f"{corpus_name}.{bt_lang}.tmp")
    orig_tmp = os.path.join(corpus_root, f"{corpus_name}.{orig_lang}.tmp")
    os.makedirs(corpus_root, exist_ok=True)
    bt_counts = Counter()
    filter_stats = FilterStats()
    with open(bt_tmp, "wt") as fout_bt, open(orig_tmp, "wt") as fout_orig:
        for orig_in, bt_in in zip(orig_ins, bt_ins):
            with open(orig_in, "rt") as fin_orig, open(bt_in, "rt") as fin_bt:
                for orig, bt in zip(fin_orig, fin_bt):
                    orig = orig.strip()
                    bt = bt.strip()
                    filter_stats.total += 1
                    decoded_pair = try_decode_pair(
                        bt, orig, spm_model, bt_lang, filter_stats, args
                    )
                    if decoded_pair is not None:
                        bt, orig = decoded_pair
                        bt_hash = xxhash.xxh3_64_intdigest(bt)
                        bt_counts[bt_hash] += 1
                        if args.max_repeat < 0 or bt_counts[bt_hash] <= args.max_repeat:
                            fout_bt.write(f"{bt}\n")
                            fout_orig.write(f"{orig}\n")
                            filter_stats.kept += 1
                        else:
                            filter_stats.duplicate_bt_filtered += 1
    bt_out = os.path.join(corpus_root, f"{corpus_name}.{bt_lang}.gz")
    orig_out = os.path.join(corpus_root, f"{corpus_name}.{orig_lang}.gz")
    if args.detruecase:
        os.system(f"{args.detruecase} < {bt_tmp} | gzip > {bt_out}")
        os.system(f"{args.detruecase} < {orig_tmp} | gzip > {orig_out}")
    else:
        os.system(f"gzip < {bt_tmp} > {bt_out}")
        os.system(f"gzip < {orig_tmp} > {orig_out}")
    os.remove(bt_tmp)
    os.remove(orig_tmp)

    print(filter_stats)
    return f"{bt_lang}-{orig_lang}", filter_stats


def main(args):
    log_folder = os.path.join(args.base_folder, "executor_logs")
    executor = AutoExecutor(
        log_folder, cluster="slurm" if not args.local_run else "local"
    )
    os.makedirs(log_folder, exist_ok=True)
    executor.update_parameters(
        slurm_partition=args.slurm_partition,
        timeout_min=args.slurm_timeout,
        nodes=1,
    )

    jobs = []
    for direction in args.directions:
        bt_lang, orig_lang = direction.split("-")
        jobs.append(executor.submit(process_lang, bt_lang, orig_lang, args))

    results = [job.result() for job in jobs]
    for direction, filter_stats in results:
        # Direction x-y here represent y data backtranslated into x.
        kept_pct = {"kept_pct": round(100 * filter_stats.kept / filter_stats.total, 2)}
        print(yaml.dump({direction: dict(filter_stats.__dict__, **kept_pct)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract backtranslations from the outputs of MOSES."
    )
    parser.add_argument(
        "--directions",
        required=True,
        nargs="*",
        help="Space-separated list of directions. Format: ${bt_lang}-${orig_lang}.",
    )
    parser.add_argument(
        "--max-copy-ratio",
        type=float,
        default=0.3,
        help="Maximum ratio of tokens that were copied from the original: "
        "len(bt_toks & orig_toks) / len(bt_toks).",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=20,
        help="Maximum number of times the same backtranslation can appear "
        "in a corpus – ignore after that.",
    )
    parser.add_argument(
        "--spm-decode",
        type=str,
        help="Path to the SPM model to be used for decoding.",
    )
    parser.add_argument(
        "--detruecase",
        type=str,
        help="Location of the MOSES detruecaser script.",
    )
    parser.add_argument(
        "--local-run",
        action="store_true",
        help="Run locally instead of on SLURM.",
    )
    parser.add_argument(
        "--slurm-partition",
        type=str,
    )
    parser.add_argument(
        "--slurm-timeout",
        type=int,
        default=1440,
    )
    parser.add_argument("--corpus-name", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument(
        "base_folder",
        type=str,
        help="Base folder. MOSES inputs and outputs are expected in "
        "{src_sharded,tgt_sharded}/${direction}/${lang}.{000..999}. "
        "Filtered outputs will be placed in corpora/$direction/bt.${lang}.gz.",
    )

    args = parser.parse_args()
    main(args)
