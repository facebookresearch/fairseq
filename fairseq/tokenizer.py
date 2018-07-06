# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import re

import torch


def build_tokenizer(args, max_length=None):
    if args.tokenizer_name == 'default':
        tokenizer_tool = Tokenizer(max_length)
    elif args.tokenizer_name == 'nltk':
        tokenizer_tool = NLTKTokenizer(max_length)
    elif args.tokenizer_name == 'sacremoses':
        tokenizer_tool = SacreMosesTokenizer(max_length)
    else:
        raise ValueError('Unknown tokenizer name: {}'.format(args.tokenizer_name))
    return tokenizer_tool

SPACE_NORMALIZER = re.compile("\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class Tokenizer(object):
    def __init__(self, max_length=None):
        self.tokenize_fn = tokenize_line
        self.max_length = max_length

    def add_file_to_dictionary(filename, dict):
        with open(filename, 'r') as f:
            for line in f:
                words = self.tokenize_line(line)

                for word in words:
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    def binarize(filename, dict, consumer, append_eos=True, reverse_order=False):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            for line in f:
                ids = self.tokenize(
                    line=line,
                    dict=dict,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1

                consumer(ids)
                ntok += len(ids)
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': len(replaced)}

    def tokenize(line, dict, add_if_not_exist=True, consumer=None, 
                 append_eos=True, reverse_order=False):
        words = self.tokenize_line(line)

        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index
        return ids

    def tokenize_line(line):
        return self.tokenize_fn(line)[:self.max_length]


class NLTKTokenizer(Tokenizer):
  def __init__(self, max_length=None):
    super().__init__(max_length)

    try:
      import nltk
      self.tokenize_fn = nltk.word_tokenize
    except ImportError as e:
      import sys
      sys.stderr.write('ERROR: Please install nltk to use.')
      raise e

    # punkt is necessary to use word_tokenize.
    if not nltk.downloader.Downloader().is_installed('punkt'):
      nltk.download('punkt')


def SacreMosesTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(max_length)

        try:
            import sacremoses
            self.tokenize_fn = sacremoses.MosesTokenizer().tokenize
        except ImportError as e:
            import sys
            sys.stderr.write('ERROR: Please install sacremoses to use.')
            raise e