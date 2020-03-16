#!/usr/bin/env python3

import os
import re
from typing import Dict, List, Optional, Set

from fairseq.data import dictionary

# TODO(T55884145): Replace with
# from fvcore.common.file_io import PathManager
from fairseq.file_io import PathManager

MAX_SPECIAL_TOKENS = 100

# Number of Byte indices is always fixed at 256 (0-255). The additional 5 indices
# correpsond to the special tokens for byte numberization including
# padding, start and end of word, start and end of sentence. These are
# separate from the special tokens in the dict and match up with the indices
# used by pre-trained ELMo.
NUM_BYTE_INDICES = 261

# Aligns with https://fburl.com/diffusion/h4s29tlr
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4


TAGS = [
    "@DIGITS",
    "@EMOTICON",
    "@FBENTITY",
    "@MULTIPUNCT",
    "@NOTRANSLATE",
    "@PERSON",
    "@PLAIN",
    "@URL",
    "@USERNAME",
]

SPACE_NORMALIZER = re.compile(r"\s+")


def default_dictionary_path(save_dir: str, dialect: str) -> str:
    return os.path.join(save_dir, f"dict.{dialect}.txt")


def default_char_dictionary_path(save_dir: str, dialect: str) -> str:
    return os.path.join(save_dir, f"char-dict.{dialect}.txt")


def tokenize_line(line, embed_bytes=False):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def char_tokenize_line(line):
    words = tokenize_line(line)
    chars = []
    for word in words:
        if word in TAGS:
            chars.append(word)
        else:
            chars.extend(c for c in word)
    return chars


def add_file_to_dictionary(filename, dict, tokenize):
    with PathManager.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            for word in tokenize(line):
                dict.add_symbol(word)
            dict.add_symbol(dict.eos_word)


class Dictionary(dictionary.Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad: str = "<pad>",
        eos: str = "</s>",
        unk: str = "<unk>",
        bos: str = "<s>",
        max_special_tokens: int = MAX_SPECIAL_TOKENS,
    ) -> None:
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols: List[str] = []
        self.count: List[int] = []
        self.indices: Dict[str, int] = {}
        self.lexicon_indices: Set[int] = set()

        self.pad_index = self.add_symbol(pad)
        assert self.pad_index == PAD_ID

        # Adds a junk symbol for vocab_constants' GO_ID
        self.add_symbol("<reserved>")

        self.eos_index = self.add_symbol(eos)
        assert self.eos_index == EOS_ID

        self.unk_index = self.add_symbol(unk)
        assert self.unk_index == UNK_ID
        self.bos_index = self.add_symbol(bos)

        # Adds junk symbols to pad up to the number of special tokens.
        num_reserved = max_special_tokens - len(self.symbols)
        for i in range(num_reserved):
            self.add_symbol(f"<reserved_{i}>")

        self.nspecial = len(self.symbols)
        assert self.nspecial == max_special_tokens

    def lexicon_indices_list(self) -> List[int]:
        return list(self.lexicon_indices)

    @classmethod
    def build_vocab_file(
        cls,
        corpus_files: List[str],
        vocab_file: str,
        max_vocab_size: int,
        tokens_with_penalty: Optional[str] = None,
        is_char_vocab: bool = False,
        embed_bytes: bool = False,
        padding_factor: int = 8,
    ) -> "Dictionary":  # https://www.python.org/dev/peps/pep-0484/#forward-references
        d = cls()

        tokenize = char_tokenize_line if is_char_vocab else tokenize_line
        embed_bytes = embed_bytes and is_char_vocab

        # if we are embedding byte ids then no need to add these to the dict
        # the ids an be obtained directly from the character
        if not embed_bytes:
            for corpus_file in corpus_files:
                add_file_to_dictionary(filename=corpus_file, dict=d, tokenize=tokenize)

        # Set indices to receive penalty
        if tokens_with_penalty:
            # Assume input tokens are unique
            lexicon = []
            with PathManager.open(tokens_with_penalty, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) == 1:
                        lexicon.append(tokens[0])

            for token, token_index in d.indices.items():
                if token in lexicon:
                    d.lexicon_indices.add(token_index)

        nwords = -1 if max_vocab_size <= 0 else max_vocab_size + d.nspecial
        d.finalize(nwords=nwords, padding_factor=padding_factor)
        d.save(vocab_file)
        print(f"Generated new vocab file saved at {vocab_file}.")
        if max_vocab_size < 0:
            print("No maximum vocab sized enforced.")
        else:
            print(f"Maximum vocab size {max_vocab_size}")
        return d

    @classmethod
    def build_vocab_file_if_nonexistent(
        cls,
        corpus_files: List[str],
        vocab_file: str,
        max_vocab_size: int,
        tokens_with_penalty: Optional[str] = None,
        is_char_vocab: bool = False,
        embed_bytes: bool = False,
        padding_factor: int = 8,
    ) -> "Dictionary":  # https://www.python.org/dev/peps/pep-0484/#forward-references
        if PathManager.isfile(vocab_file):
            d = cls.load(vocab_file)
            print(
                f"Re-using existing vocab file {vocab_file}. Specified "
                f"max vocab size of {max_vocab_size} may not be enforced."
            )
            return d

        print(
            f"Vocab file {vocab_file} does not exist. "
            "Creating new vocab file at that path."
        )
        return cls.build_vocab_file(
            corpus_files=corpus_files,
            vocab_file=vocab_file,
            max_vocab_size=max_vocab_size,
            tokens_with_penalty=tokens_with_penalty,
            is_char_vocab=is_char_vocab,
            embed_bytes=embed_bytes,
            padding_factor=padding_factor,
        )


class CharDictionary(Dictionary):
    """Character vocab with its additonal special tokens."""

    def __init__(self, word_delim="<space>", **kwargs):
        super().__init__(**kwargs)
        self.word_delim = word_delim
        self.bow_index = self.add_symbol("<bow>")
        self.eow_index = self.add_symbol("<eow>")
        self.word_delim_index = self.add_symbol(word_delim)
        self.nspecial += 3


class MaxVocabDictionary(Dictionary):
    """This dictionary takes the form of the largest dictionary supplied via push()."""

    def push(self, d: Dictionary):
        if len(d) > len(self):
            self.copy_from(d)

    def copy_from(self, d: dictionary.Dictionary):
        """Makes self a shallow copy of d."""
        self.unk_word = d.unk_word
        self.pad_word = d.pad_word
        self.eos_word = d.eos_word
        self.symbols = d.symbols
        self.count = d.count
        self.indices = d.indices
        self.pad_index = d.pad_index
        self.eos_index = d.eos_index
        self.unk_index = d.unk_index
        self.nspecial = d.nspecial
        self.lexicon_indices = d.lexicon_indices
