#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import os


class InputExample:
    def __init__(self, paragraph, qa_list, label):
        self.paragraph = paragraph
        self.qa_list = qa_list
        self.label = label


def get_examples(data_dir, set_type):
    """
    Extract paragraph and question-answer list from each json file
    """
    examples = []

    levels = ["middle", "high"]
    set_type_c = set_type.split('-')
    if len(set_type_c) == 2:
        levels = [set_type_c[1]]
        set_type = set_type_c[0]
    for level in levels:
        cur_dir = os.path.join(data_dir, set_type, level)
        for filename in os.listdir(cur_dir):
            cur_path = os.path.join(cur_dir, filename)
            with open(cur_path, 'r') as f:
                cur_data = json.load(f)
                answers = cur_data["answers"]
                options = cur_data["options"]
                questions = cur_data["questions"]
                context = cur_data["article"].replace("\n", " ")
                for i in range(len(answers)):
                    label = ord(answers[i]) - ord("A")
                    qa_list = []
                    question = questions[i]
                    for j in range(4):
                        option = options[i][j]
                        if "_" in question:
                            qa_cat = question.replace("_", option)
                        else:
                            qa_cat = " ".join([question, option])
                        qa_list.append(qa_cat)
                    examples.append(InputExample(context, qa_list, label))

    return examples


def main():
    """
    Helper script to extract paragraphs questions and answers from RACE datasets.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        help='input directory for downloaded RACE dataset',
    )
    parser.add_argument(
        "--output-dir",
        help='output directory for extracted data',
    )
    args = parser.parse_args()

    for set_type in ["train", "dev", "test-middle", "test-high"]:
        examples = get_examples(args.input_dir, set_type)
        qa_file_paths = [args.output_dir + set_type + ".input" + str(i + 1) for i in range(4)]
        qa_files = [open(qa_file_path, 'w') for qa_file_path in qa_file_paths]
        outf_context_path = args.output_dir + set_type + ".input0"
        outf_label_path = args.output_dir + set_type + ".label"
        outf_context = open(outf_context_path, 'w')
        outf_label = open(outf_label_path, 'w')
        for example in examples:
            outf_context.write(example.paragraph + '\n')
            for i in range(4):
                qa_files[i].write(example.qa_list[i] + '\n')
            outf_label.write(str(example.label) + '\n')
        
        for f in qa_files:
            f.close()
        outf_label.close()
        outf_context.close()


if __name__ == '__main__':
    main()
