# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data import Dictionary


class HFBertDictionary(Dictionary):
    """
    Dictionary for Hugginface BERT. This is using totally the same dictionary as
    Google's released bert. It doesn't have special tokens since they are included
    in dictionary file
    """

    def __init__(
        self, pad="[PAD]", unk="[UNK]", cls="[CLS]", mask="[MASK]", sep="[SEP]"
    ):
        (
            self.pad_word,
            self.unk_word,
            self.cls_word,
            self.mask_word,
            self.sep_word,
            self.eos_word,
            self.bos_word,
        ) = (
            pad,
            unk,
            cls,
            mask,
            sep,
            sep,
            sep,
        )
        self.symbols = []
        self.count = []
        self.indices = {}
        self.nspecial = 0

    def bos(self):
        """Helper to get index of bos symbol"""
        idx = self.add_symbol(self.bos_word)
        return idx

    def pad(self):
        """Helper to get index of pad symbol"""
        idx = self.add_symbol(self.pad_word)
        return idx

    def eos(self):
        """Helper to get index of eos symbol"""
        idx = self.add_symbol(self.eos_word)
        return idx

    def unk(self):
        """Helper to get index of unk symbol"""
        idx = self.add_symbol(self.unk_word)
        return idx

    def cls(self):
        """Helper to get index of cls symbol"""
        idx = self.add_symbol(self.cls_word)
        return idx

    def sep(self):
        """Helper to get index of sep symbol"""
        idx = self.add_symbol(self.sep_word)
        return idx

    def mask(self):
        """Helper to get index of sep symbol"""
        idx = self.add_symbol(self.mask_word)
        return idx
