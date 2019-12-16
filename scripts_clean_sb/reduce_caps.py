"""
This script lowercases sentences that are completely written in caps (don't contain 3 small letters in a row).
Acronyms and sentences only partially written in caps remain unchanged.
Lowercasing of caps sentences is important for the language detection.
Author: Damyana Gateva
"""

import os
import re





def process_file_lc(inF, outF):
    pattern = re.compile('[a-zäöüß]{3,}')
    patt2 = re.compile('[A-ZÄÖÜ]')

    curr = 0
    lower = 0

    with open(inF) as f, open(outF, 'w') as fOUT:
        for line in f:
            curr += 1
            #line = line.strip()

            if (re.search(pattern, line) == None) and (re.search(patt2, line) != None):
                line = line.lower()
                lower +=1
            fOUT.write(line)
        print("Lowercased {} sentences in {}".format(lower, inF))



