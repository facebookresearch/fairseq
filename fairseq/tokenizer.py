# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import re
import torch

from fairseq import dictionary


def tokenize_line(line):
    line = re.sub(r"\t", "", line)
    line = re.sub(r"^\s+", "", line)
    line = re.sub(r"\s+$", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.split()


class Tokenizer:

    @staticmethod
    def build_dictionary(filename, tokenize=tokenize_line):
        dict = dictionary.Dictionary()
        Tokenizer.add_file_to_dictionary(filename, dict, tokenize)
        dict.finalize()
        return dict

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, 'r') as f:
            for line in f.readlines():
                for word in tokenize(line):
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line):
        nseq, ntok, nunk = 0, 0, 0
        replaced = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                words = tokenize(line)
                nwords = len(words)
                ids = torch.IntTensor(nwords + 1)
                nseq = nseq + 1
                for i in range(0, len(words)):
                    word = words[i]
                    idx = dict.index(word)
                    if idx == dict.unk_index and word != dict.unk_word:
                        nunk = nunk + 1
                        if word in replaced:
                            replaced[word] = replaced[word] + 1
                        else:
                            replaced[word] = 1
                    ids[i] = idx

                ids[nwords] = dict.eos_index
                consumer(ids)
                ntok = ntok + len(ids)
        return {'nseq': nseq, 'nunk': nunk, 'ntok': ntok, 'replaced': len(replaced)}

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True):
        words = tokenize(line)
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1)
        for i in range(0, len(words)):
            if add_if_not_exist:
                ids[i] = dict.add_symbol(words[i])
            else:
                ids[i] = dict.index(words[i])
        ids[nwords] = dict.eos_index
        return ids
