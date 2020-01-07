"""This script deletes all duplicate sentences in a parallel corpus of two languages, L1 and L2. A sentence is considered
to be duplicate if the same string appears in L1 in two lines and if in the same two lines in L2, the strings are also the
 same (with regard to each other only in L2). This means if the L1 corpus e.g. contains the
same string in lines 3 and 15, but the L2 corpus contains two different strings in the respective lines (due to different translations)
 the sentences will NOT be deleted."""
# Author: Sophie Henning

import os
import sys
from time import time
from multiprocessing import Pool


class FastElimDupl:
    def __init__(self, path_to_L1_file, path_to_L2_file, num_threads):
        self.l1 = path_to_L1_file
        self.l2 = path_to_L2_file
        self.num_threads = num_threads
        self.string_to_dupl_set = dict()  # key: duplicate string, value: set of all its IDs
        self.dupl_id_to_string = dict()  # key: duplicate ID, value: corresponding duplicate string
        self.ids_to_delete = set()

    @staticmethod
    def get_dupl_ids(path_to_file):
        duplicate_file = path_to_file + '.duplicates'
        # Use command-line tools to find the duplicates
        os.system('nl ' + path_to_file + ' | sort -k 2 | uniq -D -f 1 | sort -k 2 > ' + duplicate_file)
        string_to_dupl_set = dict()  # key: duplicate string, value: set of all its IDs
        dupl_id_to_string = dict()  # key: duplicate ID, value: corresponding duplicate string

        with open(duplicate_file, "r") as f:
            for line in f:
                if line.strip() == "":  # there are some empty lines in the output of the nl command
                    continue
                line_number, string, string2 = line.split("\t")
                if string not in string_to_dupl_set.keys():
                    string_to_dupl_set[string] = set()
                string_to_dupl_set[string].add(int(line_number))
                dupl_id_to_string[int(line_number)] = string

        return string_to_dupl_set, dupl_id_to_string


    def make_new_file(self, input_path, output_path):
        line_number = 0
        with open(input_path, "r") as inp:
            with open(output_path, "w") as outp:
                for line in inp:
                    line_number += 1  # nl starts counting line numbers with 1
                    if line_number not in self.dupl_id_to_string:
                        outp.write(line)

    def run(self):
        starting_time = time()
        # Use command-line tools to create the bilingual file
        bilingual_file = self.l1 + '.bil'
        os.system('paste ' + self.l1 + ' ' + self.l2 +'  > ' + bilingual_file)


        print("Getting duplicates with sort and uniq...")
        self.string_to_dupl_set, self.dupl_id_to_string = self.get_dupl_ids(bilingual_file)
        print("Done after {:.1f} seconds.".format(time()-starting_time))

        starting_time = time()
        print("Compiling new L1 file...")
        output_l1 = self.l1[:-2] + "dupl_rem." + self.l1[-2:]
        self.make_new_file(self.l1, output_l1)
        print("Done after {:.1f} seconds.".format(time() - starting_time))
        starting_time = time()
        print("Compiling new L2 file...")
        output_l2 = self.l2[:-2] + "dupl_rem." + self.l2[-2:]
        self.make_new_file(self.l2, output_l2)
        print("Done after {:.1f} seconds.".format(time() - starting_time))
        return output_l1, output_l2

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 fast_elim_dupl_paste.py <L1 file> <L2 file> <number of threads>")
        sys.exit(1)
    eliminator = FastElimDupl(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    eliminator.run()
