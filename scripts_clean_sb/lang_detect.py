""" Creates a copy of a parallel corpus only with the sentences identified as the
    desired language 
    Date: 29.01.2019
    Author: cristinae
    modified: 18.11.19, daga

"""


from langdetect import DetectorFactory
from langdetect import detect
import os
import re
import sys
from time import time



class LanguageCleaning:
    def __init__(self, stem_in, stem_out, l1, l2):
        self.stem_in = stem_in
        self.stem_out = stem_out
        self.l1 = l1
        self.l2 = l2
        self.discarded = set()
        DetectorFactory.seed = 0
        self.out_files = []



    
    def guess_lang(self, inF, lang):
        outF = self.stem_out + '.lanRejected.' + lang

        num = re.compile("^[\d\s\W_]+$")
        rejected = set()
    
        with open(inF) as f, open(outF, 'w') as fOUT:
    
            for curr, line in enumerate(f,1):
                line = line.strip()
    
                reason = " "
                detected = " "
    
                if line == '':
                    reason = ">>>>> empty"
                elif (re.match(num, line) != None):
                    reason = ">>> number or nonlang"
                else:
                    try:
                        detected = detect(bytes(line, 'utf-8').decode('utf-8'))
                    except:
                        reason = ">>>>> encoding, nonlang"
    
                if (detected != lang) or (reason != " "):
                    rejected.add(curr)
                    fOUT.write( str(curr) + "\t" + detected + "\t" + reason +  "\t" + line + "\n" )
        print("Discarded: %s in %s " % (len(rejected), inF))
        self.discarded = self.discarded.union(rejected) 
    
    
    def combine_linenr(self, inf, lang):
        outf = self.stem_out + '.lanClean.' + lang
        self.out_files.append(outf)
        print("Total rejected sentences: %s " % (len(self.discarded)))
        with open(inf) as inF, open(outf, 'w') as outF:
            lines = inF.readlines()
            for curr, line in enumerate(lines, 1):
                if curr not in self.discarded:
                    outF.write(line)
        
    

    def run(self):
        for lang in [self.l1, self.l2]:
            inf = self.stem_in + "." + lang
            print("Checking sentences for language correspondence in {}".format(inf))
            starting_time = time()
            self.guess_lang(inf, lang)
            print("Done after {:.1f} seconds.".format(time() - starting_time))
        
        for lang in [self.l1, self.l2]:
            print("Writing language filtered files")
            inf = self.stem_out + "." + lang
            starting_time = time()
            self.combine_linenr(inf, lang)
            #print("Done after {:.1f} seconds.".format(time() - starting_time))

if __name__ == "__main__":
    if len(sys.argv != 5):
        print("Usage: python3 lang_detect.py <base_filename_in> <base_filename_out> <L1> <L2>")
        sys.exit(1)
    lanCleaner = LanguageCleaning(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    lanCleaner.run()


        
