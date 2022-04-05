# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import logging
import re
import time

from g2p_en import G2p

logger = logging.getLogger(__name__)

FAIL_SENT = "FAILED_SENTENCE"


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--lower-case", action="store_true")
    parser.add_argument("--do-filter", action="store_true")
    parser.add_argument("--use-word-start", action="store_true")
    parser.add_argument("--dup-vowel", default=1, type=int)
    parser.add_argument("--dup-consonant", default=1, type=int)
    parser.add_argument("--no-punc", action="store_true")
    parser.add_argument("--reserve-word", type=str, default="")
    parser.add_argument(
        "--reserve-first-column",
        action="store_true",
        help="first column is sentence id",
    )
    ###
    parser.add_argument("--parallel-process-num", default=1, type=int)
    parser.add_argument("--logdir", default="")
    args = parser.parse_args()
    return args


def process_sent(sent, g2p, res_wrds, args):
    sents = pre_process_sent(sent, args.do_filter, args.lower_case, res_wrds)
    pho_seqs = [do_g2p(g2p, s, res_wrds, i == 0) for i, s in enumerate(sents)]
    pho_seq = (
        [FAIL_SENT]
        if [FAIL_SENT] in pho_seqs
        else list(itertools.chain.from_iterable(pho_seqs))
    )
    if args.no_punc:
        pho_seq = remove_punc(pho_seq)
    if args.dup_vowel > 1 or args.dup_consonant > 1:
        pho_seq = dup_pho(pho_seq, args.dup_vowel, args.dup_consonant)
    if args.use_word_start:
        pho_seq = add_word_start(pho_seq)
    return " ".join(pho_seq)


def remove_punc(sent):
    ns = []
    regex = re.compile("[^a-zA-Z0-9 ]")
    for p in sent:
        if (not regex.search(p)) or p == FAIL_SENT:
            if p == " " and (len(ns) == 0 or ns[-1] == " "):
                continue
            ns.append(p)
    return ns


def do_g2p(g2p, sent, res_wrds, is_first_sent):
    if sent in res_wrds:
        pho_seq = [res_wrds[sent]]
    else:
        pho_seq = g2p(sent)
    if not is_first_sent:
        pho_seq = [" "] + pho_seq  # add space to separate
    return pho_seq


def pre_process_sent(sent, do_filter, lower_case, res_wrds):
    if do_filter:
        sent = re.sub("-", " ", sent)
        sent = re.sub("—", " ", sent)
    if len(res_wrds) > 0:
        wrds = sent.split()
        wrds = ["SPLIT_ME " + w + " SPLIT_ME" if w in res_wrds else w for w in wrds]
        sents = [x.strip() for x in " ".join(wrds).split("SPLIT_ME") if x.strip() != ""]
    else:
        sents = [sent]
    if lower_case:
        sents = [s.lower() if s not in res_wrds else s for s in sents]
    return sents


def dup_pho(sent, dup_v_num, dup_c_num):
    """
    duplicate phoneme defined as cmudict
    http://www.speech.cs.cmu.edu/cgi-bin/cmudict
    """
    if dup_v_num == 1 and dup_c_num == 1:
        return sent
    ns = []
    for p in sent:
        ns.append(p)
        if re.search(r"\d$", p):
            for i in range(1, dup_v_num):
                ns.append(f"{p}-{i}P")
        elif re.search(r"\w", p):
            for i in range(1, dup_c_num):
                ns.append(f"{p}-{i}P")
    return ns


def add_word_start(sent):
    ns = []
    do_add = True
    ws = "▁"
    for p in sent:
        if do_add:
            p = ws + p
            do_add = False
        if p == " ":
            do_add = True
        else:
            ns.append(p)
    return ns


def load_reserve_word(reserve_word):
    if reserve_word == "":
        return []
    with open(reserve_word, "r") as fp:
        res_wrds = [x.strip().split() for x in fp.readlines() if x.strip() != ""]
        assert sum([0 if len(x) == 2 else 1 for x in res_wrds]) == 0
        res_wrds = dict(res_wrds)
    return res_wrds


def process_sents(sents, args):
    g2p = G2p()
    out_sents = []
    res_wrds = load_reserve_word(args.reserve_word)
    for sent in sents:
        col1 = ""
        if args.reserve_first_column:
            col1, sent = sent.split(None, 1)
        sent = process_sent(sent, g2p, res_wrds, args)
        if args.reserve_first_column and col1 != "":
            sent = f"{col1} {sent}"
        out_sents.append(sent)
    return out_sents


def main():
    args = parse()
    out_sents = []
    with open(args.data_path, "r") as fp:
        sent_list = [x.strip() for x in fp.readlines()]
    if args.parallel_process_num > 1:
        try:
            import submitit
        except ImportError:
            logger.warn(
                "submitit is not found and only one job is used to process the data"
            )
            submitit = None

    if args.parallel_process_num == 1 or submitit is None:
        out_sents = process_sents(sent_list, args)
    else:
        # process sentences with parallel computation
        lsize = len(sent_list) // args.parallel_process_num + 1
        executor = submitit.AutoExecutor(folder=args.logdir)
        executor.update_parameters(timeout_min=1000, cpus_per_task=4)
        jobs = []
        for i in range(args.parallel_process_num):
            job = executor.submit(
                process_sents, sent_list[lsize * i : lsize * (i + 1)], args
            )
            jobs.append(job)
        is_running = True
        while is_running:
            time.sleep(5)
            is_running = sum([job.done() for job in jobs]) < len(jobs)
        out_sents = list(itertools.chain.from_iterable([job.result() for job in jobs]))
    with open(args.out_path, "w") as fp:
        fp.write("\n".join(out_sents) + "\n")


if __name__ == "__main__":
    main()
