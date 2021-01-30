#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import re
from collections import deque
from enum import Enum

import numpy as np


"""
    Utility modules for computation of Word Error Rate,
    Alignments, as well as more granular metrics like
    deletion, insersion and substitutions.
"""


class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4


class Token(object):
    def __init__(self, lbl="", st=np.nan, en=np.nan):
        if np.isnan(st):
            self.label, self.start, self.end = "", 0.0, 0.0
        else:
            self.label, self.start, self.end = lbl, st, en


class AlignmentResult(object):
    def __init__(self, refs, hyps, codes, score):
        self.refs = refs  # std::deque<int>
        self.hyps = hyps  # std::deque<int>
        self.codes = codes  # std::deque<Code>
        self.score = score  # float


def coordinate_to_offset(row, col, ncols):
    return int(row * ncols + col)


def offset_to_row(offset, ncols):
    return int(offset / ncols)


def offset_to_col(offset, ncols):
    return int(offset % ncols)


def trimWhitespace(str):
    return re.sub(" +", " ", re.sub(" *$", "", re.sub("^ *", "", str)))


def str2toks(str):
    pieces = trimWhitespace(str).split(" ")
    toks = []
    for p in pieces:
        toks.append(Token(p, 0.0, 0.0))
    return toks


class EditDistance(object):
    def __init__(self, time_mediated):
        self.time_mediated_ = time_mediated
        self.scores_ = np.nan  # Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
        self.backtraces_ = (
            np.nan
        )  # Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> backtraces_;
        self.confusion_pairs_ = {}

    def cost(self, ref, hyp, code):
        if self.time_mediated_:
            if code == Code.match:
                return abs(ref.start - hyp.start) + abs(ref.end - hyp.end)
            elif code == Code.insertion:
                return hyp.end - hyp.start
            elif code == Code.deletion:
                return ref.end - ref.start
            else:  # substitution
                return abs(ref.start - hyp.start) + abs(ref.end - hyp.end) + 0.1
        else:
            if code == Code.match:
                return 0
            elif code == Code.insertion or code == Code.deletion:
                return 3
            else:  # substitution
                return 4

    def get_result(self, refs, hyps):
        res = AlignmentResult(refs=deque(), hyps=deque(), codes=deque(), score=np.nan)

        num_rows, num_cols = self.scores_.shape
        res.score = self.scores_[num_rows - 1, num_cols - 1]

        curr_offset = coordinate_to_offset(num_rows - 1, num_cols - 1, num_cols)

        while curr_offset != 0:
            curr_row = offset_to_row(curr_offset, num_cols)
            curr_col = offset_to_col(curr_offset, num_cols)

            prev_offset = self.backtraces_[curr_row, curr_col]

            prev_row = offset_to_row(prev_offset, num_cols)
            prev_col = offset_to_col(prev_offset, num_cols)

            res.refs.appendleft(curr_row - 1)  # Note: this was .push_front() in C++
            res.hyps.appendleft(curr_col - 1)
            if curr_row - 1 == prev_row and curr_col == prev_col:
                res.codes.appendleft(Code.deletion)
            elif curr_row == prev_row and curr_col - 1 == prev_col:
                res.codes.appendleft(Code.insertion)
            else:
                # assert(curr_row - 1 == prev_row and curr_col - 1 == prev_col)
                ref_str = refs[res.refs[0]].label
                hyp_str = hyps[res.hyps[0]].label

                if ref_str == hyp_str:
                    res.codes.appendleft(Code.match)
                else:
                    res.codes.appendleft(Code.substitution)

                    confusion_pair = "%s -> %s" % (ref_str, hyp_str)
                    if confusion_pair not in self.confusion_pairs_:
                        self.confusion_pairs_[confusion_pair] = 1
                    else:
                        self.confusion_pairs_[confusion_pair] += 1

            curr_offset = prev_offset

        return res

    def align(self, refs, hyps):
        if len(refs) == 0 and len(hyps) == 0:
            return np.nan

        # NOTE: we're not resetting the values in these matrices because every value
        # will be overridden in the loop below. If this assumption doesn't hold,
        # be sure to set all entries in self.scores_ and self.backtraces_ to 0.
        self.scores_ = np.zeros((len(refs) + 1, len(hyps) + 1))
        self.backtraces_ = np.zeros((len(refs) + 1, len(hyps) + 1))

        num_rows, num_cols = self.scores_.shape

        for i in range(num_rows):
            for j in range(num_cols):
                if i == 0 and j == 0:
                    self.scores_[i, j] = 0.0
                    self.backtraces_[i, j] = 0
                    continue

                if i == 0:
                    self.scores_[i, j] = self.scores_[i, j - 1] + self.cost(
                        None, hyps[j - 1], Code.insertion
                    )
                    self.backtraces_[i, j] = coordinate_to_offset(i, j - 1, num_cols)
                    continue

                if j == 0:
                    self.scores_[i, j] = self.scores_[i - 1, j] + self.cost(
                        refs[i - 1], None, Code.deletion
                    )
                    self.backtraces_[i, j] = coordinate_to_offset(i - 1, j, num_cols)
                    continue

                # Below here both i and j are greater than 0
                ref = refs[i - 1]
                hyp = hyps[j - 1]
                best_score = self.scores_[i - 1, j - 1] + (
                    self.cost(ref, hyp, Code.match)
                    if (ref.label == hyp.label)
                    else self.cost(ref, hyp, Code.substitution)
                )

                prev_row = i - 1
                prev_col = j - 1
                ins = self.scores_[i, j - 1] + self.cost(None, hyp, Code.insertion)
                if ins < best_score:
                    best_score = ins
                    prev_row = i
                    prev_col = j - 1

                delt = self.scores_[i - 1, j] + self.cost(ref, None, Code.deletion)
                if delt < best_score:
                    best_score = delt
                    prev_row = i - 1
                    prev_col = j

                self.scores_[i, j] = best_score
                self.backtraces_[i, j] = coordinate_to_offset(
                    prev_row, prev_col, num_cols
                )

        return self.get_result(refs, hyps)


class WERTransformer(object):
    def __init__(self, hyp_str, ref_str, verbose=True):
        self.ed_ = EditDistance(False)
        self.id2oracle_errs_ = {}
        self.utts_ = 0
        self.words_ = 0
        self.insertions_ = 0
        self.deletions_ = 0
        self.substitutions_ = 0

        self.process(["dummy_str", hyp_str, ref_str])

        if verbose:
            print("'%s' vs '%s'" % (hyp_str, ref_str))
            self.report_result()

    def process(self, input):  # std::vector<std::string>&& input
        if len(input) < 3:
            print(
                "Input must be of the form <id> ... <hypo> <ref> , got ",
                len(input),
                " inputs:",
            )
            return None

        # Align
        # std::vector<Token> hyps;
        # std::vector<Token> refs;

        hyps = str2toks(input[-2])
        refs = str2toks(input[-1])

        alignment = self.ed_.align(refs, hyps)
        if alignment is None:
            print("Alignment is null")
            return np.nan

        # Tally errors
        ins = 0
        dels = 0
        subs = 0
        for code in alignment.codes:
            if code == Code.substitution:
                subs += 1
            elif code == Code.insertion:
                ins += 1
            elif code == Code.deletion:
                dels += 1

        # Output
        row = input
        row.append(str(len(refs)))
        row.append(str(ins))
        row.append(str(dels))
        row.append(str(subs))
        # print(row)

        # Accumulate
        kIdIndex = 0
        kNBestSep = "/"

        pieces = input[kIdIndex].split(kNBestSep)

        if len(pieces) == 0:
            print(
                "Error splitting ",
                input[kIdIndex],
                " on '",
                kNBestSep,
                "', got empty list",
            )
            return np.nan

        id = pieces[0]
        if id not in self.id2oracle_errs_:
            self.utts_ += 1
            self.words_ += len(refs)
            self.insertions_ += ins
            self.deletions_ += dels
            self.substitutions_ += subs
            self.id2oracle_errs_[id] = [ins, dels, subs]
        else:
            curr_err = ins + dels + subs
            prev_err = np.sum(self.id2oracle_errs_[id])
            if curr_err < prev_err:
                self.id2oracle_errs_[id] = [ins, dels, subs]

        return 0

    def report_result(self):
        # print("----------  Summary ---------------")
        if self.words_ == 0:
            print("No words counted")
            return

        # 1-best
        best_wer = (
            100.0
            * (self.insertions_ + self.deletions_ + self.substitutions_)
            / self.words_
        )

        print(
            "\tWER = %0.2f%% (%i utts, %i words, %0.2f%% ins, "
            "%0.2f%% dels, %0.2f%% subs)"
            % (
                best_wer,
                self.utts_,
                self.words_,
                100.0 * self.insertions_ / self.words_,
                100.0 * self.deletions_ / self.words_,
                100.0 * self.substitutions_ / self.words_,
            )
        )

    def wer(self):
        if self.words_ == 0:
            wer = np.nan
        else:
            wer = (
                100.0
                * (self.insertions_ + self.deletions_ + self.substitutions_)
                / self.words_
            )
        return wer

    def stats(self):
        if self.words_ == 0:
            stats = {}
        else:
            wer = (
                100.0
                * (self.insertions_ + self.deletions_ + self.substitutions_)
                / self.words_
            )
            stats = dict(
                {
                    "wer": wer,
                    "utts": self.utts_,
                    "numwords": self.words_,
                    "ins": self.insertions_,
                    "dels": self.deletions_,
                    "subs": self.substitutions_,
                    "confusion_pairs": self.ed_.confusion_pairs_,
                }
            )
        return stats


def calc_wer(hyp_str, ref_str):
    t = WERTransformer(hyp_str, ref_str, verbose=0)
    return t.wer()


def calc_wer_stats(hyp_str, ref_str):
    t = WERTransformer(hyp_str, ref_str, verbose=0)
    return t.stats()


def get_wer_alignment_codes(hyp_str, ref_str):
    """
    INPUT: hypothesis string, reference string
    OUTPUT: List of alignment codes (intermediate results from WER computation)
    """
    t = WERTransformer(hyp_str, ref_str, verbose=0)
    return t.ed_.align(str2toks(ref_str), str2toks(hyp_str)).codes


def merge_counts(x, y):
    # Merge two hashes which have 'counts' as their values
    # This can be used for example to merge confusion pair counts
    #   conf_pairs = merge_counts(conf_pairs, stats['confusion_pairs'])
    for k, v in y.items():
        if k not in x:
            x[k] = 0
        x[k] += v
    return x
