"""
Modified from https://github.com/keithito/tacotron
"""


import string
import re
from unidecode import unidecode
from .numbers import normalize_numbers
from typing import List, Callable


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
}


def expand_abbreviations(text, lang="en"):
    if lang not in _abbreviations:
        return text

    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text, lang="en"):
    return normalize_numbers(text, lang)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


PUNCTUATIONS_EXCLUDE_APOSTROPHE = (
    string.punctuation.replace("'", "") + "¡¨«°³º»¿‘“”…♪♫ˆᵉ™，ʾ˚"
)
PUNCTUATIONS_TO_SPACE = "-/–·—•"


def remove_punctuations(text, punctuations=string.punctuation):
    text = text.translate(
        str.maketrans(PUNCTUATIONS_TO_SPACE, " " * len(PUNCTUATIONS_TO_SPACE))
    )
    return text.translate(str.maketrans("", "", punctuations))


def remove_parentheses(text: str) -> str:
    # remove all substring within () or []
    out = ""
    num_p = 0
    start_i = 0
    for i, c in enumerate(text):
        if c == "(" or c == "[":
            if num_p == 0 and i > start_i:
                out += text[start_i:i]
            num_p += 1
        elif c == ")" or c == "]":
            num_p -= 1
            if num_p == 0:
                start_i = i + 1

    if len(text) > start_i:
        out += text[start_i:]

    return out.strip()


REMAP_CHARS = {
    "`": "'",
    "’ ": " ",
    "’": "'",
}


def remap_chars(text, remap_chars=REMAP_CHARS):
    for k, v in remap_chars.items():
        text = text.replace(k, v)
    return text


def expand_capitals(text):
    words = text.split()
    for i, w in enumerate(words):
        if w.isupper():
            words[i] = " ".join(w)
    return " ".join(words)


def english_cleaners(text, punctuations=string.punctuation):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = remap_chars(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = remove_parentheses(text)
    text = remove_punctuations(text, punctuations=punctuations)
    text = collapse_whitespace(text)
    text = text.strip()
    return text


def english_cleaners_keep_apostrophe(text):
    return english_cleaners(text, punctuations=PUNCTUATIONS_EXCLUDE_APOSTROPHE)


def fisher_text_cleaners(text):
    # remove the convert_to_ascii cleaner to keep Spanish characters
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = remove_punctuations(text, punctuations=PUNCTUATIONS_EXCLUDE_APOSTROPHE)
    text = collapse_whitespace(text)
    return text


def text_cleaners(text, lang="en"):
    text = remap_chars(text)
    text = expand_capitals(text)
    text = lowercase(text)
    text = remove_parentheses(text)

    if lang in ["en", "fr", "es"]:
        text = expand_numbers(text, lang)
    text = expand_abbreviations(text, lang)
    text = remove_punctuations(text, punctuations=PUNCTUATIONS_EXCLUDE_APOSTROPHE)
    text = collapse_whitespace(text)
    text = text.strip()
    return text


def apply_text_functions(text_funcs: List[Callable], text: str) -> str:
    for func in text_funcs:
        text = func(text)
    return text
