# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.data import Dictionary


class MaskedLMDictionary(Dictionary):
    """
    Dictionary for Masked Language Modelling tasks. This extends Dictionary by
    adding the mask symbol.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
    ):
        super().__init__(pad, eos, unk)
        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        self.nspecial = len(self.symbols)

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index


class BertDictionary(MaskedLMDictionary):
    """
    Dictionary for BERT task. This extends MaskedLMDictionary by adding support
    for cls and sep symbols.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
        cls='<cls>',
        sep='<sep>'
    ):
        super().__init__(pad, eos, unk, mask)
        self.cls_word = cls
        self.sep_word = sep
        self.cls_index = self.add_symbol(cls)
        self.sep_index = self.add_symbol(sep)
        self.nspecial = len(self.symbols)

    def cls(self):
        """Helper to get index of cls symbol"""
        return self.cls_index

    def sep(self):
        """Helper to get index of sep symbol"""
        return self.sep_index
