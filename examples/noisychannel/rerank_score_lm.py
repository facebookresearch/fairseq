# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from fairseq import options

from . import rerank_options, rerank_utils


def score_lm(args):
    using_nbest = args.nbest_list is not None
    (
        pre_gen,
        left_to_right_preprocessed_dir,
        right_to_left_preprocessed_dir,
        backwards_preprocessed_dir,
        lm_preprocessed_dir,
    ) = rerank_utils.get_directories(
        args.data_dir_name,
        args.num_rescore,
        args.gen_subset,
        args.gen_model_name,
        args.shard_id,
        args.num_shards,
        args.sampling,
        args.prefix_len,
        args.target_prefix_frac,
        args.source_prefix_frac,
    )

    predictions_bpe_file = pre_gen + "/generate_output_bpe.txt"
    if using_nbest:
        print("Using predefined n-best list from interactive.py")
        predictions_bpe_file = args.nbest_list

    gen_output = rerank_utils.BitextOutputFromGen(
        predictions_bpe_file, bpe_symbol=args.post_process, nbest=using_nbest
    )

    if args.language_model is not None:
        lm_score_file = rerank_utils.rescore_file_name(
            pre_gen, args.prefix_len, args.lm_name, lm_file=True
        )

    if args.language_model is not None and not os.path.isfile(lm_score_file):
        print("STEP 4.5: language modeling for P(T)")
        if args.lm_bpe_code is None:
            bpe_status = "no bpe"
        elif args.lm_bpe_code == "shared":
            bpe_status = "shared"
        else:
            bpe_status = "different"

        rerank_utils.lm_scoring(
            lm_preprocessed_dir,
            bpe_status,
            gen_output,
            pre_gen,
            args.lm_dict,
            args.lm_name,
            args.language_model,
            args.lm_bpe_code,
            128,
            lm_score_file,
            args.target_lang,
            args.source_lang,
            prefix_len=args.prefix_len,
        )


def cli_main():
    parser = rerank_options.get_reranking_parser()
    args = options.parse_args_and_arch(parser)
    score_lm(args)


if __name__ == "__main__":
    cli_main()
