#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
import shutil
import sys


pt_regexp = re.compile(r"checkpoint(\d+|_\d+_\d+|_[a-z]+)\.pt")
pt_regexp_epoch_based = re.compile(r"checkpoint(\d+)\.pt")
pt_regexp_update_based = re.compile(r"checkpoint_\d+_(\d+)\.pt")


def parse_checkpoints(files):
    entries = []
    for f in files:
        m = pt_regexp_epoch_based.fullmatch(f)
        if m is not None:
            entries.append((int(m.group(1)), m.group(0)))
        else:
            m = pt_regexp_update_based.fullmatch(f)
            if m is not None:
                entries.append((int(m.group(1)), m.group(0)))
    return entries


def last_n_checkpoints(files, n):
    entries = parse_checkpoints(files)
    return [x[1] for x in sorted(entries, reverse=True)[:n]]


def every_n_checkpoints(files, n):
    entries = parse_checkpoints(files)
    return [x[1] for x in sorted(sorted(entries)[::-n])]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively delete checkpoint files from `root_dir`, "
            "but preserve checkpoint_best.pt and checkpoint_last.pt"
        )
    )
    parser.add_argument("root_dirs", nargs="*")
    parser.add_argument(
        "--save-last", type=int, default=0, help="number of last checkpoints to save"
    )
    parser.add_argument(
        "--save-every", type=int, default=0, help="interval of checkpoints to save"
    )
    parser.add_argument(
        "--preserve-test",
        action="store_true",
        help="preserve checkpoints in dirs that start with test_ prefix (default: delete them)",
    )
    parser.add_argument(
        "--delete-best", action="store_true", help="delete checkpoint_best.pt"
    )
    parser.add_argument(
        "--delete-last", action="store_true", help="delete checkpoint_last.pt"
    )
    parser.add_argument(
        "--no-dereference", action="store_true", help="don't dereference symlinks"
    )
    args = parser.parse_args()

    files_to_desymlink = []
    files_to_preserve = []
    files_to_delete = []
    for root_dir in args.root_dirs:
        for root, _subdirs, files in os.walk(root_dir):
            if args.save_last > 0:
                to_save = last_n_checkpoints(files, args.save_last)
            else:
                to_save = []
            if args.save_every > 0:
                to_save += every_n_checkpoints(files, args.save_every)
            for file in files:
                if not pt_regexp.fullmatch(file):
                    continue
                full_path = os.path.join(root, file)
                if (
                    not os.path.basename(root).startswith("test_") or args.preserve_test
                ) and (
                    (file == "checkpoint_last.pt" and not args.delete_last)
                    or (file == "checkpoint_best.pt" and not args.delete_best)
                    or file in to_save
                ):
                    if os.path.islink(full_path) and not args.no_dereference:
                        files_to_desymlink.append(full_path)
                    else:
                        files_to_preserve.append(full_path)
                else:
                    files_to_delete.append(full_path)

    if len(files_to_desymlink) == 0 and len(files_to_delete) == 0:
        print("Nothing to do.")
        sys.exit(0)

    files_to_desymlink = sorted(files_to_desymlink)
    files_to_preserve = sorted(files_to_preserve)
    files_to_delete = sorted(files_to_delete)

    print("Operations to perform (in order):")
    if len(files_to_desymlink) > 0:
        for file in files_to_desymlink:
            print(" - preserve (and dereference symlink): " + file)
    if len(files_to_preserve) > 0:
        for file in files_to_preserve:
            print(" - preserve: " + file)
    if len(files_to_delete) > 0:
        for file in files_to_delete:
            print(" - delete: " + file)
    while True:
        resp = input("Continue? (Y/N): ")
        if resp.strip().lower() == "y":
            break
        elif resp.strip().lower() == "n":
            sys.exit(0)

    print("Executing...")
    if len(files_to_desymlink) > 0:
        for file in files_to_desymlink:
            realpath = os.path.realpath(file)
            print("rm " + file)
            os.remove(file)
            print("cp {} {}".format(realpath, file))
            shutil.copyfile(realpath, file)
    if len(files_to_delete) > 0:
        for file in files_to_delete:
            print("rm " + file)
            os.remove(file)


if __name__ == "__main__":
    main()
