#!/usr/bin/env python3 -u

import argparse
import fileinput
import logging
import os
import sys

from fairseq.models.transformer import TransformerModel


logging.getLogger().setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--en2fr", required=True, help="path to en2fr model")
    parser.add_argument(
        "--fr2en", required=True, help="path to fr2en mixture of experts model"
    )
    parser.add_argument(
        "--user-dir", help="path to fairseq examples/translation_moe/src directory"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=10,
        help="(keep at 10 unless using a different model)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=["-"],
        help='input files to paraphrase; "-" for stdin',
    )
    args = parser.parse_args()

    if args.user_dir is None:
        args.user_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # examples/
            "translation_moe",
            "src",
        )
        if os.path.exists(args.user_dir):
            logging.info("found user_dir:" + args.user_dir)
        else:
            raise RuntimeError(
                "cannot find fairseq examples/translation_moe/src "
                "(tried looking here: {})".format(args.user_dir)
            )

    logging.info("loading en2fr model from:" + args.en2fr)
    en2fr = TransformerModel.from_pretrained(
        model_name_or_path=args.en2fr,
        tokenizer="moses",
        bpe="sentencepiece",
    ).eval()

    logging.info("loading fr2en model from:" + args.fr2en)
    fr2en = TransformerModel.from_pretrained(
        model_name_or_path=args.fr2en,
        tokenizer="moses",
        bpe="sentencepiece",
        user_dir=args.user_dir,
        task="translation_moe",
    ).eval()

    def gen_paraphrases(en):
        fr = en2fr.translate(en)
        return [
            fr2en.translate(fr, inference_step_args={"expert": i})
            for i in range(args.num_experts)
        ]

    logging.info("Type the input sentence and press return:")
    for line in fileinput.input(args.files):
        line = line.strip()
        if len(line) == 0:
            continue
        for paraphrase in gen_paraphrases(line):
            print(paraphrase)


if __name__ == "__main__":
    main()
