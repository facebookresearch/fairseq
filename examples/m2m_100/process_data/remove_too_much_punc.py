import gzip
import argparse
from string import punctuation

def len_no_punc(s, punc):
    return len([ch for ch in s if ch in punc])

def filter_overpunc(len_npunc, len_sen):
    return len_npunc < 0.5*len_sen

def main(args):
    punc = punctuation + "—|–"
    print('Processing file {}'.format(args.input))
    with gzip.open(args.input, 'rt', encoding=args.encoding) as tsv:
        with open(args.bitext + '.' + args.src_lang, 'wt', encoding=args.encoding) as fsrc:
            with open(args.bitext + '.' + args.tgt_lang, 'wt', encoding=args.encoding) as ftgt:
                line = tsv.readline()
                if not line:
                    continue
                fields = line.split('\t')

                src, tgt = fields[1], fields[2]

                nchar_npunc_src = len_no_punc(src, punc)
                nchar_npunc_tgt = len_no_punc(tgt, punc)

                if filter_overpunc(nchar_npunc_src, len(src)) and filter_overpunc(nchar_npunc_tgt, len(tgt)):
                    fsrc.write(src.strip() + '\n')
                    ftgt.write(tgt.strip() + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument('--encoding', default='utf-8', help='character encoding for input/output')
    parser.add_argument('--bitext', type=str, required=True, help='language direction')
    parser.add_argument('--src-lang', type=str, required=True, help='Source language')
    parser.add_argument('--tgt-lang', type=str, required=True, help='Target language')
    main(parser.parse_args())
