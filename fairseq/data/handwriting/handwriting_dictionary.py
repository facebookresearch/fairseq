

from .alphabet import Alphabet


class HandwritingDictionary(Alphabet):

    def __init__(
        self, 
        alphabet_file=None, 
        #*,  # begin keyword-only arguments
        bos="{",
        pad="~",
        eos="}",
        unk="@",  # in Alphabet
        ):  #extra_special_symbols=None,):

        # [!] bos, pad, eos etc. need to be in dict file
        super().__init__(alphabet_file, unk=(unk,))
        #self._alphabet = Alphabet(alphabet_file, unk=(unk,))  
        for c, descr in zip((bos, pad, eos, unk), ("bos", "pad", "eos", "unk")):
            if not self.existDict(c):
                print('WARNING:', descr, 'token', c, 'not in vocab')
        self.bos_char, self.unk_char, self.pad_char, self.eos_char = bos, unk, pad, eos
        #self.symbols = []
        #self.count = []
        self.indices = {}
        self.bos_index = self.ch2idx(bos)  #self.add_symbol(bos)
        self.pad_index = self.ch2idx(pad)  #self.add_symbol(pad)
        self.eos_index = self.ch2idx(eos)  #self.add_symbol(eos)
        #self.unk_index = self.ch2idx(unk)  #self.add_symbol(unk)
        # if extra_special_symbols:
        #     for s in extra_special_symbols:
        #         self.add_symbol(s)
        #self.nspecial = len(self.symbols)

    # TODO here and in other places - not sure what is actually needed
    # def __eq__(self, other):
    #     return self.indices == other.indices

    def __getitem__(self, idx):
        #if idx < len(self.symbols):
        #    return self.symbols[idx]
        return self.idx2ch(idx)  #unk_word

    # len defined in Alphabet

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        # if sym in self.indices:
        #     return self.indices[sym]
        # return self.unk_index
        return self.ch2idx(sym)

    # string() almost double with Alphabet stuff
    def string(
        self,
        tensor,
        # bpe_symbol=None,
        # escape_unk=False,
        # extra_symbols_to_ignore=None,
        # unk_string=None,
    ):
        return self.idx2str(tensor)  # should be ok, needs only iter on tensor

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    # moved to Alphabet
    # def unk(self):
    #     """Helper to get index of unk symbol"""
    #     return self.unk_index

    def encode_line(self):
        pass # TODO NOT SURE IF NEEDED OR JUST DO/IS IN DATASET, if needed this needs to return list of indexes

    # 
    # CTC criterion seems to have 1234567 additional hidden params that are to-be-used-with-words hardcoded    



