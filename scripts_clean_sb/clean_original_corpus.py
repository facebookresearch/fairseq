"""Removes sentences whose letters are mainly non-Latin in both languages in the original corpora (threshold: 0.5) and unescapes HTML character entities such as &#160.
 If a sentence contains non-Latin characters in one of the languages, it will be removed in both languages."""
# Author: Sophie Henning

import regex  # https://pypi.org/project/regex/
import html
import argparse
from time import time


class Cleaner:
    def __init__(self, path_to_l1, path_to_l2, fix_latin=True, fix_html=True, write_sentences=False):
        self.l1 = path_to_l1
        self.l2 = path_to_l2
        self.non_latin = set()  # set containing all the line numbers that contain mainly non-Latin characters in L1 or L2
        self.fix_latin = fix_latin  # indicates whether all non-Latin character sentences should be removed
        self.fix_html = fix_html
        self.write_sentences = write_sentences
        self.sentences = list()

    @staticmethod
    def get_length_of_tokens_in_list(tokens_list):
        len_string = 0
        for token in tokens_list:
            len_string += len(token)
        return len_string

    def find_non_latin(self, path_to_language):
        with open(path_to_language, "r") as f:
            line_number = 0
            pattern = regex.compile(r'[^\p{Latin}\p{posix_punct}\p{BasicLatin}\p{InBasic_Latin}\p{InLatin-1_Supplement}'
                                    r'\p{InLatin_Extended-A}\p{InLatin_Extended-B}\p{InLatin_Extended_Additional}\u200a'
                                    r'\u200b\u200c\u200d\u200e\u200f]+')
            for line in f:
                match_list = regex.findall(pattern, line)  # get all occurrences of non-Latin characters
                len_non_latin = self.get_length_of_tokens_in_list(match_list)
                len_line = self.get_length_of_tokens_in_list(line.split())  # length of tokens in line, excluding whitespace
                # we don't want empty lines
                if len_line < 1:
                    self.non_latin.add(line_number)
                else:
                    ratio = len_non_latin/len_line
                    # only delete line if more than half of its characters are non-Latin
                    if ratio > 0.5:
                       self.non_latin.add(line_number)
                line_number += 1

    def make_new_file(self, path_to_language):
        new_file_name = path_to_language[:-3]
        if self.fix_latin:
            new_file_name += ".fl"
        if self.fix_html:
            new_file_name += ".fh"
        new_file_name += path_to_language[-3:]
        with open(path_to_language, "r") as inp:
            with open(new_file_name, "w") as outp:
                line_number = 0
                for line in inp:
                    if line_number in self.non_latin:  # empty if fix_latin==False
                        line_number += 1
                        if self.write_sentences:  # store the sentences that were removed for debugging purposes
                            self.sentences.append(line)
                        continue
                    if self.fix_html:
                        # replace ampquot; with &quot; - otherwise html.unescape doesn't recognize it
                        line = regex.sub(r'ampquot;', r'&quot;', line)
                        # use built-in unescape for the HTML character entities
                        line = html.unescape(line)  # https://docs.python.org/3/library/html.html#html.unescape
                    if line.strip() != "":  # don't write entirely empty lines
                        outp.write(line)
                    line_number += 1
        return new_file_name

    def run(self):
        if self.fix_latin:
            starting_time = time()
            print("Finding sentences with non-Latin characters in L1... ")
            self.find_non_latin(self.l1)
            print("Done after {:.1f} seconds.".format(time() - starting_time))
            starting_time = time()
            print("Finding sentences with non-Latin characters in L2... ")
            self.find_non_latin(self.l2)
            print("Done after {:.1f} seconds.".format(time() - starting_time))

        starting_time = time()
        print("Compiling new L1 file... ")
        new_l1_file = self.make_new_file(self.l1)
        print("Done after {:.1f} seconds.".format(time() - starting_time))
        starting_time = time()
        print("Compiling new L2 file... ")
        new_l2_file = self.make_new_file(self.l2)
        print("Done after {:.1f} seconds.".format(time() - starting_time))

        if self.write_sentences:
            print("Writing removed sentences to file...")
            with open(self.l1[:-3] + ".removed_sentences", "w") as f:
                for line in self.sentences:
                    f.write(line)

        return new_l1_file, new_l2_file


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Clean the original parallel corpora in Moses format")
    argparser.add_argument("L1_file", type=str, help="Path of L1 corpus file (mandatory)")
    argparser.add_argument("L2_file", type=str, help="Path of L2 corpus file (mandatory)")
    argparser.add_argument("-nl", "--no-latin", dest="do_latin", default=True, action='store_false',
                           help='Do not remove sentences containing non-Latin characters')
    argparser.add_argument("-nh", "--no-html", dest="do_html", default=True, action='store_false',
                           help='Do not unescape HTML character entities')
    argparser.add_argument("-ws", "--write-sentences", dest="write_sentences", default=False, action='store_true',
                           help='Write all removed sentences to file (for checking purposes)')
    args = argparser.parse_args()

    cleaner = Cleaner(args.L1_file, args.L2_file, args.do_latin, args.do_html, args.write_sentences)
    cleaner.run()
