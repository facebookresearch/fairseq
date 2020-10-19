#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
extract text from json reddit dataset
"""

import argparse
import gzip
import json
import os
import sys


REPLACE_MAP = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    args = parser.parse_args()

    def extract_text(json):
        text = ""

        def try_add(key):
            if (
                key in json
                and json[key] != ""
                and json[key] != "[deleted]"
                and json[key] != "[removed]"
            ):
                return json[key] + "\n"
            else:
                return ""

        text += try_add("title")
        text += try_add("selftext")
        text += try_add("body")

        if "children" in json:
            for c in json["children"]:
                text += extract_text(c)

        return text

    for filename in os.listdir(args.source_dir):
        if not filename.endswith(".jsonl.gz"):
            print(f"skipping{filename}", file=sys.stderr)
            continue

        p = os.path.join(args.source_dir, filename)
        print("processing " + p, file=sys.stderr)

        with gzip.GzipFile(p, "r") as fin:
            json_bytes = fin.read()

        json_strs = filter(None, json_bytes.decode("utf-8").split("\n"))
        for js in json_strs:
            data = json.loads(js)
            text = extract_text(data)
            for k, v in REPLACE_MAP.items():
                text = text.replace(k, v)
            if len(text) > 0:
                print(text)


if __name__ == "__main__":
    main()
