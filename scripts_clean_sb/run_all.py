"""Runs all preprocessing scripts, assuming that clean_original_corpus.py and fast_elim_dupl_multi.py lie in the same directory
Requires Moses scripts: https://github.com/moses-smt/mosesdecoder/tree/master/scripts
For language identification requires installation of lang_detect: https://pypi.org/project/langdetect/
"""
# Author: Sophie Henning
# modified: Damyana Gateva 29.11.2019
import clean_original_corpus
import fast_elim_dupl_paste
import reduce_caps 
import lang_detect 
import os
import argparse
from time import time


class RunAll:
    def __init__(self, path_to_Moses_scripts, path_to_models, path_to_corpus, l1_code, l2_code, num_threads, tidy_up,
    caps_lower, lang_detect):
        self.moses = path_to_Moses_scripts
        self.models = path_to_models  # sth like models/modelTC.EpWP.en (la_code will be stripped automatically)
        self.corpus_path = path_to_corpus  # it suffices to give the path of one corpus, for retrieving, l1_code and l2_code will be used
        self.l1 = l1_code  # de, es, fr, en
        self.l2 = l2_code
        self.stem_proc = ""
        self.stem = self.get_stem(path_to_corpus)
        self.l1_path = self.stem + "." + self.l1 
        self.l2_path = self.stem + "." + self.l2 
        self.threads = num_threads
        self.tidy_up = tidy_up
        self.caps_lower = caps_lower
        self.lang_detect = lang_detect
    
    def get_stem(self, filepath):
        ending = filepath.split(".")[-1]
        return filepath[:-(len(ending) + 1)]  
    

    """Removes all sentences containing non-Latin characters and unescapes HTML character entities"""
    def fix_latin_html(self):
        cleaner = clean_original_corpus.Cleaner(self.l1_path, self.l2_path)
        return cleaner.run()

    def run_perl_scripts_norm(self, file_path, la_code):
        stem = self.stem_proc
        starting_time = time()
        print("Running normalization 1 for", la_code)
        norm1_path = stem + '.norm1.' + la_code
        norm1_cmd = 'perl ' + self.moses + 'tokenizer/replace-unicode-punctuation.perl < ' + file_path + ' > ' + norm1_path
        print("Command:", norm1_cmd)
        os.system(norm1_cmd)
        print("Done after {:.1f} seconds.".format(time() - starting_time))
        starting_time = time()
        print("Running normalization 2 for", la_code)
        norm2_path = stem + '.norm2.' + la_code
        norm2_cmd = 'perl ' + self.moses + 'tokenizer/normalize-punctuation.perl -l ' + la_code + ' < ' + norm1_path + ' > ' \
                    + norm2_path
        print("Command:", norm2_cmd)
        os.system(norm2_cmd)
        print("Done after {:.1f} seconds.".format(time() - starting_time))
        starting_time = time()
        print("Running normalization 3 for", la_code)
        norm3_path = stem + '.norm3.' + la_code
        norm3_cmd = 'perl ' + self.moses + 'tokenizer/remove-non-printing-char.perl < ' + norm2_path + ' > ' + norm3_path
        print("Command:", norm3_cmd)
        os.system(norm3_cmd)
        print("Done after {:.1f} seconds.".format(time() - starting_time))
        
        return norm3_path




    def run_perl_scripts_tok_tc(self, file_path, la_code):
        stem = self.stem_proc
        starting_time = time()
        print("Running tokenizer for", la_code)
        tok_path = stem + '.tok.' + la_code
        tok_cmd = 'perl ' + self.moses + 'tokenizer/tokenizer.perl -l ' + la_code + ' -no-escape -threads ' + \
                  str(self.threads) + ' < ' + file_path + ' > ' + tok_path
        print("Command:", tok_cmd)
        os.system(tok_cmd)
        print("Done after {:.1f} seconds.".format(time() - starting_time))
        starting_time = time()
        print("Running truecaser for", la_code)
        tc_path = stem + '.tc.' + la_code
        tc_cmd = 'perl ' + self.moses + 'recaser/truecase.perl --model ' + self.models[:-2] + la_code + ' < ' + tok_path\
                 + ' > ' + tc_path
        print("Command:", tc_cmd)
        os.system(tc_cmd)
        print("Done after {:.1f} seconds.".format(time() - starting_time))
        return tc_path



    def caps_to_lower(self, file_path, la_code):
        stem = self.stem_proc
        starting_time = time() 
        lower_path = stem + '.lowerCaps.' + la_code
        print("Lowercasing sentences written only in Capitals")
        reduce_caps.process_file_lc(file_path, lower_path)
        print("Done after {:.1f} seconds.".format(time() - starting_time))
        return lower_path
    

    
    def filter_wrong_lang(self, l1_path, l2_path):
        stem_in = self.get_stem(l1_path)
        assert ( stem_in == self.get_stem(l2_path) ) , "Different basenames of input filepaths"
        stem_out = self.stem_proc
        lanCleaner = lang_detect.LanguageCleaning(stem_in, stem_out, self.l1, self.l2)
        lanCleaner.run()
        return lanCleaner.out_files



    def elim_dupl(self, l1_path, l2_path):
        eliminator = fast_elim_dupl_paste.FastElimDupl(l1_path, l2_path, self.threads)
        return eliminator.run()

    def tidy_up_folder(self, output_l1):
        corpus_folder = self.corpus_path[:(self.corpus_path.rfind('/')+1)]
        original_folder = corpus_folder + 'original/'  # path where all original files should be stored
        intermediate_folder = corpus_folder + 'intermediate/' # path where all intermediate results of the preprocessing are stored
        final_folder = corpus_folder + 'final/' # path where the final two corpora versions are stored
        os.system('mkdir ' + original_folder)
        os.system('mkdir ' + intermediate_folder)
        os.system('mkdir ' + final_folder)
        stem = self.corpus_path[:-2]
        os.system('mv ' + stem + self.l1 + ' ' + original_folder)
        os.system('mv ' + stem + self.l2 + ' ' + original_folder)
        #os.system('mv ' + stem + 'ids' + ' ' + original_folder)
        os.system('mv ' + output_l1[:-2] + '* ' + final_folder)
        os.system('mv ' + stem + '* ' + intermediate_folder)

    def run(self):
        starting_time = time()
        new_l1_path, new_l2_path = self.fix_latin_html()
        self.stem_proc = self.get_stem(new_l1_path)


        l1_norm = self.run_perl_scripts_norm(new_l1_path, self.l1)
        l2_norm = self.run_perl_scripts_norm(new_l2_path, self.l2)

        l1_cleaner = l1_norm
        l2_cleaner = l2_norm
        ###
        if (self.caps_lower):
            l1_cleaner = self.caps_to_lower(l1_cleaner, self.l1)
            l2_cleaner = self.caps_to_lower(l2_cleaner, self.l2)


            if (self.lang_detect):
                l1_cleaner, l2_cleaner = self.filter_wrong_lang(l1_cleaner, l2_cleaner)

        ###
        l1_tc = self.run_perl_scripts_tok_tc(l1_cleaner, self.l1)
        l2_tc = self.run_perl_scripts_tok_tc(l2_cleaner, self.l2)


        output_l1, _ = self.elim_dupl(l1_tc, l2_tc)
        if self.tidy_up:
            self.tidy_up_folder(output_l1)
        print("Total running time: {:.1f} seconds".format(time() - starting_time))

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Run all preprocessing scripts for the corpora in Moses format")
    argparser.add_argument("moses", type=str, help="Path to Moses scripts, e.g. ../scripts/ (mandatory)")
    argparser.add_argument("models", type=str, help="Path to one of the models for the Moses truecaser, e.g. ../models/modelTC.EpWP.de (mandatory)")
    argparser.add_argument("corpus", type=str, help="Path to one of the corpus files (mandatory)")
    argparser.add_argument("l1_code", type=str, help="Code of L1, e.g. de (mandatory)")
    argparser.add_argument("l2_code", type=str, help="Code of L2, e.g. es (mandatory)")
    argparser.add_argument("-nt", "--num-threads", type=int, default=1,
                           help='Number of threads that can be used, default: 1')
    argparser.add_argument("-tu", "--tidy-up", default=False, action='store_true', help='Tidy up the folder with all the corpus files afterwards')
    argparser.add_argument("-capl", "--caps-low", default=False, action='store_true', help='If the sentence is written completely in caps, lowercase it')
    argparser.add_argument("-ld", "--lang-detect", default=False, action='store_true', help='Identify the language of the sentence and delete if not corresponding')
    args = argparser.parse_args()
    if (args.lang_detect) and (not args.caps_low):
        print("Option --lang-detect goes only with option --caps-low")
        print("Setting option --caps-low to True")
        args.caps_low = True
    runner = RunAll(args.moses, args.models, args.corpus, args.l1_code, args.l2_code, args.num_threads, args.tidy_up,
    args.caps_low, args.lang_detect)
    runner.run()

