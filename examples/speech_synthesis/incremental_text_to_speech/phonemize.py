import os
import sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Phonemizer(object):
    def __init__(self, lang="en"):
        self.lang = lang

        if lang == "en":
            try:
                from g2p_en import G2p
            except ImportError:
                print("Please install g2p-en 'pip install g2p-en'")

            self.g2p = G2p()

        else:
            raise NotImplementedError("No phonemizer available for language {0}".format(lang))

    def convert_word_to_phonemes(self, raw_word):
        if self.lang == "en":
            return self.g2p(raw_word)