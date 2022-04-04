# Copyright (c) Carnegie Mellon University (Jiatong Shi)
#
# # This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import tacotron_cleaner.cleaners


def main(args):
    pre_clean = open(args.recognized_output, "r", encoding="utf-8")
    if args.output_path is not None:
        cleaned = open(args.recognized_output + ".cleaned", "w", encoding="utf-8")
    else:
        cleaned = open(args.output_path, "w", encoding="utf-8")
    if args.sort:
        utts = sorted(pre_clean.read().split("\n"))
    else:
        utts = pre_clean.read().split("\n")
    
    for utt in utts:
        _, content = utt.split(maxsplit=1)
        cleaned.write("{}".format(tacotron_cleaner.cleaners.custom_english_cleaners(content)))
    cleaned.close()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recognized_output", tyep=str, default="recognized.txt", help="output for recognition results"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="output path of recognized output"
    )
    parser.add_argument(
        "--sort", type=bool, default=False
    )
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()