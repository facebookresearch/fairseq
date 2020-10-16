# -*- coding: utf8 -*-
#   Copyright 2019 JSALT2019 Distant Supervision Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import json

class bidict(dict):
    """
    Here is a class for a bidirectional dict, inspired by Finding key from value in Python dictionary and modified to allow the following 2) and 3).

    Note that :

    1) The inverse directory bd.inverse auto-updates itself when the standard dict bd is modified.
    2) The inverse directory bd.inverse[value] is always a list of key such that bd[key] == value.
    3) Unlike the bidict module from https://pypi.python.org/pypi/bidict, here we can have 2 keys having same value, this is very important.
    """
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


class Alphabet:
    """
    The set of characters a "word" can contain.

    Args:
        filename: if specified, loads from a json file that maps characters to
            indices
        input_dict: if specified, constructs from a char -> idx dict. If
            neither filename or input_dict is specified, an empty alphabet is
            constructed
        translation_dict: if not already specified in the alphabet, can map
            together characters.
        unk: the unknown symbols.
        blank: the blank symbols.
        space: the spaces symbols.
    """
    def __init__(self, filename_=None, input_dict=None,
                 translation_dict={'_': ' '},
                 unk=("@",), blank=("*",), space=(' ', '_')):

        if filename_:
            self.chars = bidict(self.readDictionary(filename_))
            print('Alphabet constructed from', filename_,
                  'size=', len(self.chars))
        elif input_dict is not None:
            self.chars = bidict(input_dict)
            print('Alphabet constructed from dictionnary, '
                  'size=', len(self.chars))
        else:
            self.chars = bidict({
                    k: i
                    for i, chs in enumerate([blank, space, unk])
                    for k in chs
            })
            print('Alphabet constructed empty')
        for c in unk:
            if c not in self.chars:
                print('Warning: UNK token', c, 'not in vocab')
        for c in space:
            if c not in self.chars:
                print('Warning: space token', c, 'not in vocab')
        for c in blank:
            if c not in self.chars:
                print('Warning: blank token', c, 'not in vocab')
        self.translation_dict = dict(translation_dict)
        self.unk = unk
        self.blank = blank
        self.space = space

    def readDictionary(self, filename_):
        with open(filename_) as fp:
            return json.load(fp)

    def writeDictionary(self, filename_):
        with open(filename_,'w') as fp:
            json.dump(self.chars, fp)

    # Transcript util functions
    def insertDict(self, ch_, idx_):
        self.chars[ch_] = int(idx_)

    def existDict(self, ch):
        return ch in self.chars

    def find_key(self, val):
        return self.chars.inverse.get(val, self.unk[0])[0]

    def __len__(self):
        return len(self.chars)

    def ch2idx(self, c_):
        if c_ in self.translation_dict:
            c_ = self.translation_dict[c_]
        idx = self.chars.get(c_, None)
        if idx is not None:
            return int(idx)

        return self.chars[self.unk[0]]

    def idx2ch(self, idx_):
        return self.find_key(idx_)

    def symList2idxList(self, syms_):
        result = []
        for s in syms_:
            classIdx = self.ch2idx(s)
            result.append(classIdx)
        return result

    def idx2str(self, list_, noDuplicates=True, noBlanks=False,
                join_char=''):
        result = []
        for i in list_:
            ch = self.find_key(i)
            result.append(ch)
        result = join_char.join(result)

        if noDuplicates:
            result = self.removeDuplicates(result)

        if noBlanks:
            result = self.removeZerosFromString(result)

        return result

    '''
        In : a string
        Out: a string without duplicates of idx 0 (nor 1 if 1 == ' ')
    '''
    def removeDuplicates(self, in_):
        newWord = ""

        idx = 0
        for ch in in_:
            if idx <= 0:
                newWord += ch
                continue

            if ch not in self.blank and ch not in self.space:
                newWord += ch
                continue

            if ch != in_[idx-1]:
                newWord += ch
            idx += 1

        return newWord

    def removeZeros(self, list_):
        return [i for i in list_ if i != 0]

    def removeZerosFromString(self, result):
        newWord = ""
        for ch in result:
            if ch not in self.blank:
                newWord = newWord + ch
        return newWord
