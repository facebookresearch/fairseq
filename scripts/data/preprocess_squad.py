#!/usr/bin/env python3

import argparse
import json
import os
import string
import re
import sys
sys.path.append('/private/home/yinhanliu/pytorch-pretrained-BERT')
from pytorch_pretrained_bert.tokenization import BertTokenizer, whitespace_tokenize

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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def process(s):
        try:
            return tokenizer.tokenize(s)
        except:
            print('failed on', s)
            raise

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False


    for inp in args.inputs:
        bad_qs = 0
        num_qs = 0
        filename = os.path.basename(inp)
        base_filename = os.path.splitext(filename)[0]
        s1_filename = base_filename + '_1.txt'
        s2_filename = base_filename + '_2.txt'
        s3_filename = base_filename + '_3.txt'
        s4_filename = base_filename + '_4.txt'
        id_filename = base_filename + '.id'
        label_filename = base_filename + '.lbl'
        with open(inp, 'r') as f_in, open(os.path.join(args.output, s1_filename), 'w') as s1_out, open(os.path.join(args.output, s2_filename), 'w') as s2_out, open(os.path.join(args.output, id_filename), 'w') as id_out, open(os.path.join(args.output, label_filename), 'w') as lbl_out, open(os.path.join(args.output, s3_filename), 'w') as s3_out,open(os.path.join(args.output, s4_filename), 'w') as s4_out:
            data = json.load(f_in)
            for example in data['data']:
                for p in example['paragraphs']:
                    context = p['context']
                    doc_tokens = []
                    char_to_word_offset = []
                    prev_is_whitespace = True
                    for c in context:
                        if is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)

                    orig_to_tok_index = []
                    tok_to_orig_index = []
                    all_doc_tokens = []
                    for (i, token) in enumerate(doc_tokens):
                        orig_to_tok_index.append(len(all_doc_tokens))
                        sub_tokens = process(token)
                        for sub_token in sub_tokens:
                            tok_to_orig_index.append(i)
                            all_doc_tokens.append(sub_token)

                    for qa in p['qas']:
                        num_qs += 1
                        q = process(qa['question'])
                        is_impossible = True #qa['is_impossible']
                        answer = qa['answers'][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                        tok_start_position = orig_to_tok_index[start_position]
                        if end_position < len(doc_tokens) - 1:
                            tok_end_position = orig_to_tok_index[end_position + 1] - 1
                        else:
                            tok_end_position = len(all_doc_tokens) - 1

                        (tok_start_position, tok_end_position) = _improve_answer_span(
                             all_doc_tokens, tok_start_position, tok_end_position, process,
                             orig_answer_text)
                        if not is_impossible:
                            # print('bad question:', str(q))
                            bad_qs += 1
                            continue
                        print(' '.join(all_doc_tokens), file=s1_out)
                        print(' '.join(q), file=s2_out)
                        print(' '.join(doc_tokens), file=s3_out)
                        print(' '.join([str(ii) for ii in tok_to_orig_index]), file=s4_out)
                        print(qa['id'], file=id_out)
                        lbl_str = f'{int(is_impossible)}'
                        lbl_str += f' {tok_start_position} {tok_end_position}'
                        print(lbl_str, file=lbl_out)
        print('bad questions:', bad_qs, 'out of', num_qs)


def _improve_answer_span(doc_tokens, input_start, input_end, process,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(process(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


if __name__ == '__main__':
    main()
