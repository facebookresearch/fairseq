# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample_rate", type=int, default=16_000)

    args = parser.parse_args()

    with open(args.manifest, "r") as manifest, open(args.units, "r") as units, open(
        args.output, "w"
    ) as outp:
        root = manifest.readline().strip()
        root = pathlib.Path(root)

        for manifest_line, unit_line in zip(manifest.readlines(), units.readlines()):
            path, frames = manifest_line.split()
            duration = int(frames) / float(args.sample_rate)
            fname = root / path
            speaker = fname.parent.parent.name

            units = unit_line.split("|")[1]

            print(
                json.dumps(
                    dict(
                        audio=str(root / path),
                        duration=duration,
                        hubert_km100=units.strip(),
                        speaker=speaker,
                    )
                ),
                file=outp,
            )


if __name__ == "__main__":
    main()
