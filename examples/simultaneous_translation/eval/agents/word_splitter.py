class SubwordSplitter(object):
    def process_line(self, string):
        raise NotImplementedError

    def split(self, string: str) -> list:
        raise NotImplementedError

class NoneWordSplitter(object):
    def __init__(self, model):
        pass

    def split(self, string):
        return [string]
    
    def process_line(self, string):
        return [string]

    def finished_word(self, string):
        return True

    def merge(self, list_of_string):
        return "".join(list_of_string)

    def last_full_word_step(self, tokens, step):
        return len(tokens)

    def end_idx_last_full_word(self, tokens):
        return len(tokens)

class BPESubwordSplitter(object):
    def __init__(self, args):
        super().__init__()
        from subword_nmt.apply_bpe import BPE
        with open(args.model) as f:
            self.model = BPE(f)
    
    def split(self, string: str) -> list:
        return self.model.EncodeAsPieces(string)
    
    def end_idx_last_full_word(self, tokens):
        length = len(tokens)
        # Begin of word indices
        bow_indices = [i for i, t in enumerate(tokens) if t[0] == '\u2581']

        if len(bow_indices) < 2:
            return 0
        else:
            return bow_indices[-1]
    
    def merge(self, list_of_string):
        return self.model.DecodePieces(list_of_string)

class SentencePieceModelWordSplitter(object):
    def __init__(self, model_path):
        super().__init__()
        import sentencepiece as spm
        self.model = spm.SentencePieceProcessor()
        self.model.Load(model_path)
    
    def split(self, string: str) -> list:
        return self.model.EncodeAsPieces(string)
    
    def end_idx_last_full_word(self, tokens):
        length = len(tokens)
        # Begin of word indices
        bow_indices = [i for i, t in enumerate(tokens) if t[0] == '\u2581']

        if len(bow_indices) < 2:
            return 0
        else:
            return bow_indices[-1]
    
    def merge(self, list_of_string):
        return self.model.DecodePieces(list_of_string)

