# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math
import torch


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""
    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>'):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.add_symbol('<Lua heritage>')
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.nspecial = len(self.symbols)

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor):
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            sentences = [self.string(line) for line in tensor]
            return '\n'.join(sentences)

        eos = self.eos()
        return ' '.join([self[i] for i in tensor if i != eos])

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def finalize(self):
        """Sort symbols by frequency in descending order, ignoring special ones."""
        self.count, self.symbols = zip(
            *sorted(zip(self.count, self.symbols),
                    key=(lambda x: math.inf if self.indices[x[1]] < self.nspecial else x[0]),
                    reverse=True)
        )

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @staticmethod
    def load(f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """

        if isinstance(f, str):
            with open(f, 'r') as fd:
                return Dictionary.load(fd)

        d = Dictionary()
        for line in f.readlines():
            idx = line.rfind(' ')
            word = line[:idx]
            count = int(line[idx+1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d

    def save(self, f, threshold=3, nwords=-1):
        """Stores dictionary into a text file"""
        if isinstance(f, str):
            with open(f, 'w') as fd:
                return self.save(fd, threshold, nwords)
        cnt = 0
        for i, t in enumerate(zip(self.symbols, self.count)):
            if i >= self.nspecial and t[1] >= threshold \
                    and (nwords < 0 or cnt < nwords):
                print('{} {}'.format(t[0], t[1]), file=f)
                cnt += 1
