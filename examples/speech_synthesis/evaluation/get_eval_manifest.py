# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import csv
from pathlib import Path


def main(args):
    """
    `uid syn ref text`
    """
    in_root = Path(args.generation_root).resolve()
    ext = args.audio_format
    with open(args.audio_manifest) as f, open(args.output_path, "w") as f_out:
        reader = csv.DictReader(
            f, delimiter="\t", quotechar=None, doublequote=False,
            lineterminator="\n", quoting=csv.QUOTE_NONE
        )
        header = ["id", "syn", "ref", "text", "speaker"]
        f_out.write("\t".join(header) + "\n")
        for row in reader:
            dir_name = f"{ext}_{args.sample_rate}hz_{args.vocoder}"
            id_ = row["id"]
            syn = (in_root / dir_name / f"{id_}.{ext}").as_posix()
            ref = row["audio"]
            if args.use_resynthesized_target:
                ref = (in_root / f"{dir_name}_tgt" / f"{id_}.{ext}").as_posix()
            sample = [id_, syn, ref, row["tgt_text"], row["speaker"]]
            f_out.write("\t".join(sample) + "\n")
    print(f"wrote evaluation file to {args.output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-root",  help="output directory for generate_waveform.py"
    )
    parser.add_argument(
        "--audio-manifest",
        help="used to determine the original utterance ID and text"
    )
    parser.add_argument(
        "--output-path", help="path to output evaluation spec file"
    )
    parser.add_argument(
        "--use-resynthesized-target", action="store_true",
        help="use resynthesized reference instead of the original audio"
    )
    parser.add_argument("--vocoder", type=str, default="griffin_lim")
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--audio-format", type=str, default="wav")
    args = parser.parse_args()

    main(args)
