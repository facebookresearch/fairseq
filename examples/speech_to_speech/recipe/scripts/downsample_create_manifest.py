# Copyright (c) Carnegie Mellon University (Jiatong Shi)
#
# # This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import librosa
import logging
import os


def main(args):
    for subset in args.subsets:
        target_dir = os.path.join(args.tgt, subset)
        source_dir = os.path.join(args.src, subset)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        manifest = open(os.path.join(args.tgt, "manifest_{}.tsv".format(subset)), "w", encoding="utf-8")
        manifest.write("{}\n".format(os.path.abspath(args.tgt)))

        if not os.path.exists(source_dir):
            logging.warning("source dir {} not exist".format(source_dir))
        file_list = os.listdir(source_dir)
        for i, fname in enumerate(file_list):
            info, sr = librosa.load("{}/{}".format(source_dir, fname))
            # use sox for reformating
            os.system("sox {}/{} -c 1 -t wavpcm -r {} {}/{}".format(source_dir, fname, args.fs, target_dir, fname))
            manifest.write("{}/{}\t{}\n".format(target_dir, fname, int(len(info) * sr / args.fs)))
            if subset in  ["dev", "valid", "development", "validation"] and i == 500:
                break # limit dev size
        manifest.close()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", type=str, required=True, help="source audio directory (for downsample and manifest generation)"
    )
    parser.add_argument(
        "--tgt", type=str, required=True, help="target downsampled audio directory"
    )
    parser.add_argument(
        "--subset", type=str, required=True, help="subset", action="append"
    )
    parser.add_argument(
        "--fs", type=int, help="target sampling rate", default=16000
    )
    parser.add_argument(
        "--dev_size", type=int, help="# of dev set", default=500
    )
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()