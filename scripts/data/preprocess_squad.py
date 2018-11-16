#!/usr/bin/env python3

import argparse
import json
import mosestokenizer as mt
import os
import string
import re

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--inputs',
        required=True,
        nargs='+',
        help='files to process.',
    )
    parser.add_argument(
        '--output',
        required=True,
        metavar='DIR',
        help='Path for output',
    )

    args = parser.parse_args()
    print(args)

    tokenize = mt.MosesTokenizer('en', old_version=True)

    def process(s):
        try:
            s = s.replace('\n', '')
            # hacks
            s = re.sub(r'(\d+)\-(\d+)', r'\1 - \2', s)
            return tokenize(s.strip())
        except:
            print('failed on', s)
            raise

    # from https://codereview.stackexchange.com/questions/19627/finding-sub-list
    def find(haystack, needle):
        """Return the index at which the sequence needle appears in the
        sequence haystack, or -1 if it is not found, using the Boyer-
        Moore-Horspool algorithm. The elements of needle and haystack must
        be hashable.

        >>> find([1, 1, 2], [1, 2])
        1

        """
        h = len(haystack)
        n = len(needle)
        skip = {needle[i]: n - i - 1 for i in range(n - 1)}
        i = n - 1
        while i < h:
            for j in range(n):
                if haystack[i - j] != needle[-j - 1]:
                    i += skip.get(haystack[i], n)
                    break
            else:
                return i - n + 1
        return -1

    def remove_trailing_punc(text):
        exclude = set(string.punctuation)
        no_punc = text
        while len(no_punc) > 0 and no_punc[-1] in exclude:
            no_punc = no_punc[:-1]
        if len(no_punc) == 0:
            return text
        else:
            return no_punc

    for inp in args.inputs:
        bad_qs = 0
        num_qs = 0
        filename = os.path.basename(inp)
        base_filename = os.path.splitext(filename)[0]
        s1_filename = base_filename + '_1.txt'
        s2_filename = base_filename + '_2.txt'
        label_filename = base_filename + '.lbl'
        with open(inp, 'r') as f_in, open(os.path.join(args.output, s1_filename), 'w') as s1_out, open(
                os.path.join(args.output, s2_filename), 'w') as s2_out, open(os.path.join(args.output, label_filename),
                                                                             'w') as lbl_out:
            data = json.load(f_in)
            for example in data['data']:
                for p in example['paragraphs']:
                    context = [remove_trailing_punc(t) for t in process(p['context'])]
                    for qa in p['qas']:
                        num_qs += 1
                        q = process(qa['question'])
                        is_impossible = qa['is_impossible']
                        spans = []
                        for a in qa['answers']:
                            text = [remove_trailing_punc(t) for t in process(a['text'])]
                            match = find(context, text)

                            if match == -1:
                                if len(text) == 1:
                                    for i, c in enumerate(context):
                                        if c.startswith(text[0]) or c.endswith(text[0]):
                                            match = i
                                            break

                            if match == -1:
                                # print('could not find ' + str(text) + ' in ' + str(context))
                                continue
                            spans.append((match, match + len(text)))
                        if not is_impossible and len(spans) == 0:
                            # print('bad question:', str(q))
                            bad_qs += 1
                            continue
                        print(' '.join(context), file=s1_out)
                        print(' '.join(q), file=s2_out)
                        lbl_str = f'{int(is_impossible)}'
                        for s in spans:
                            lbl_str += f' {s[0]} {s[1]}'
                        print(lbl_str, file=lbl_out)
        print('bad questions:', bad_qs, 'out of', num_qs)


if __name__ == '__main__':
    main()
