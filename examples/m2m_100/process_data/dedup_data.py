import argparse
from collections import namedtuple
import os

DATADIR = "/path/to/train_data"
DEDUP_FROM_DIR = "/path/to/eval/data"
OUTPUT_DIR = "/path/to/output/data"


def main(args):
    languages = set()
    for language_directory in os.listdir(DATADIR):
        if "_" in language_directory:
            src, tgt = language_directory.split("_")
            languages.add(LanguagePair(src=src, tgt=tgt))

    data = existing_data()
    train_languages = sorted(languages)
    for language_pair in train_languages[args.start_index:args.start_index + args.size]:
        print(language_pair)
        dedup(language_pair, data)


LanguagePair = namedtuple("LanguagePair", ["src", "tgt"])


def existing_data():
    data = set()
    for file in os.listdir(DEDUP_FROM_DIR):
        with open(os.path.join(DEDUP_FROM_DIR, file)) as f:
            data |= set(f.readlines())
    return data
 
def dedup(language_pair, data, verbose=True, output=True):
    train_filenames = LanguagePair(
            src=f"{DATADIR}/{language_pair.src}_{language_pair.tgt}/train.{language_pair.src}",
            tgt=f"{DATADIR}/{language_pair.src}_{language_pair.tgt}/train.{language_pair.tgt}",
        )

    output_filenames = LanguagePair(
        src=f"{OUTPUT_DIR}/train.dedup.{language_pair.src}-{language_pair.tgt}.{language_pair.src}",
        tgt=f"{OUTPUT_DIR}/train.dedup.{language_pair.src}-{language_pair.tgt}.{language_pair.tgt}"
    )

    # If output exists, skip this pair. It has already been done.
    if (os.path.exists(output_filenames.src) and
        os.path.exists(output_filenames.tgt)):
        if verbose:
            print(f"{language_pair.src}-{language_pair.tgt} already done.")
        return

    if verbose:
        print(f"{language_pair.src}-{language_pair.tgt} ready, will check dups.")

    # If there is no output, no need to actually do the loop.
    if not output:
        return

    if os.path.exists(train_filenames.src) and os.path.exists(train_filenames.tgt):
        with open(train_filenames.src) as f:
            train_source = f.readlines()

        with open(train_filenames.tgt) as f:
            train_target = f.readlines()

        # do dedup
        new_train_source = []
        new_train_target = []
        for i, train_line in enumerate(train_source):
            if train_line not in data and train_target[i] not in data:
                new_train_source.append(train_line)
                new_train_target.append(train_target[i])

        assert len(train_source) == len(train_target)
        assert len(new_train_source) == len(new_train_target)
        assert len(new_train_source) <= len(train_source)

        with open(output_filenames.src, "w") as o:
            for line in new_train_source:
                o.write(line)

        with open(output_filenames.tgt, "w") as o:
            for line in new_train_target:
                o.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start-index", required=True, type=int)
    parser.add_argument("-n", "--size", required=True, type=int)
    main(parser.parse_args())
