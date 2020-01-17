# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import redirect_stdout
import os

from fairseq import options
from fairseq_cli import generate

from . import rerank_options, rerank_utils


def score_bw(args):
        if args.backwards1:
            scorer1_src = args.target_lang
            scorer1_tgt = args.source_lang
        else:
            scorer1_src = args.source_lang
            scorer1_tgt = args.target_lang

        if args.score_model2 is not None:
            if args.backwards2:
                scorer2_src = args.target_lang
                scorer2_tgt = args.source_lang
            else:
                scorer2_src = args.source_lang
                scorer2_tgt = args.target_lang

        rerank1_is_gen = args.gen_model == args.score_model1 and args.source_prefix_frac is None
        rerank2_is_gen = args.gen_model == args.score_model2 and args.source_prefix_frac is None

        pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, \
            backwards_preprocessed_dir, lm_preprocessed_dir = \
            rerank_utils.get_directories(args.data_dir_name, args.num_rescore, args.gen_subset,
                                         args.gen_model_name, args.shard_id, args.num_shards,
                                         args.sampling, args.prefix_len, args.target_prefix_frac,
                                         args.source_prefix_frac)

        score1_file = rerank_utils.rescore_file_name(pre_gen, args.prefix_len, args.model1_name,
                                                     target_prefix_frac=args.target_prefix_frac,
                                                     source_prefix_frac=args.source_prefix_frac,
                                                     backwards=args.backwards1)

        if args.score_model2 is not None:
            score2_file = rerank_utils.rescore_file_name(pre_gen, args.prefix_len, args.model2_name,
                                                         target_prefix_frac=args.target_prefix_frac,
                                                         source_prefix_frac=args.source_prefix_frac,
                                                         backwards=args.backwards2)

        if args.right_to_left1:
            rerank_data1 = right_to_left_preprocessed_dir
        elif args.backwards1:
            rerank_data1 = backwards_preprocessed_dir
        else:
            rerank_data1 = left_to_right_preprocessed_dir

        gen_param = ["--batch-size", str(128), "--score-reference", "--gen-subset", "train"]
        if not rerank1_is_gen and not os.path.isfile(score1_file):
            print("STEP 4: score the translations for model 1")

            model_param1 = ["--path", args.score_model1, "--source-lang", scorer1_src, "--target-lang", scorer1_tgt]
            gen_model1_param = [rerank_data1] + gen_param + model_param1

            gen_parser = options.get_generation_parser()
            input_args = options.parse_args_and_arch(gen_parser, gen_model1_param)

            with open(score1_file, 'w') as f:
                with redirect_stdout(f):
                    generate.main(input_args)

        if args.score_model2 is not None and not os.path.isfile(score2_file) and not rerank2_is_gen:
            print("STEP 4: score the translations for model 2")

            if args.right_to_left2:
                rerank_data2 = right_to_left_preprocessed_dir
            elif args.backwards2:
                rerank_data2 = backwards_preprocessed_dir
            else:
                rerank_data2 = left_to_right_preprocessed_dir

            model_param2 = ["--path", args.score_model2, "--source-lang", scorer2_src, "--target-lang", scorer2_tgt]
            gen_model2_param = [rerank_data2] + gen_param + model_param2

            gen_parser = options.get_generation_parser()
            input_args = options.parse_args_and_arch(gen_parser, gen_model2_param)

            with open(score2_file, 'w') as f:
                with redirect_stdout(f):
                    generate.main(input_args)


def cli_main():
    parser = rerank_options.get_reranking_parser()
    args = options.parse_args_and_arch(parser)
    score_bw(args)


if __name__ == '__main__':
    cli_main()
