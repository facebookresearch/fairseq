#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate n-best translations using a trained model.
"""

from contextlib import redirect_stdout
import os
import subprocess

from fairseq import options
from fairseq_cli import generate, preprocess

from . import rerank_options, rerank_utils


def gen_and_reprocess_nbest(args):
    if args.score_dict_dir is None:
        args.score_dict_dir = args.data
    if args.prefix_len is not None:
        assert args.right_to_left1 is False, "prefix length not compatible with right to left models"
        assert args.right_to_left2 is False, "prefix length not compatible with right to left models"

    if args.nbest_list is not None:
        assert args.score_model2 is None

    if args.backwards1:
        scorer1_src = args.target_lang
        scorer1_tgt = args.source_lang
    else:
        scorer1_src = args.source_lang
        scorer1_tgt = args.target_lang

    store_data = os.path.join(os.path.dirname(__file__))+"/rerank_data/"+args.data_dir_name
    if not os.path.exists(store_data):
        os.makedirs(store_data)

    pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, \
        backwards_preprocessed_dir, lm_preprocessed_dir = \
        rerank_utils.get_directories(args.data_dir_name, args.num_rescore, args.gen_subset,
                                     args.gen_model_name, args.shard_id, args.num_shards,
                                     args.sampling, args.prefix_len, args.target_prefix_frac,
                                     args.source_prefix_frac)
    assert not (args.right_to_left1 and args.backwards1), "backwards right to left not supported"
    assert not (args.right_to_left2 and args.backwards2), "backwards right to left not supported"
    assert not (args.prefix_len is not None and args.target_prefix_frac is not None), \
        "target prefix frac and target prefix len incompatible"

    # make directory to store generation results
    if not os.path.exists(pre_gen):
        os.makedirs(pre_gen)

    rerank1_is_gen = args.gen_model == args.score_model1 and args.source_prefix_frac is None
    rerank2_is_gen = args.gen_model == args.score_model2 and args.source_prefix_frac is None

    if args.nbest_list is not None:
        rerank2_is_gen = True

    # make directories to store preprossed nbest list for reranking
    if not os.path.exists(left_to_right_preprocessed_dir):
        os.makedirs(left_to_right_preprocessed_dir)
    if not os.path.exists(right_to_left_preprocessed_dir):
        os.makedirs(right_to_left_preprocessed_dir)
    if not os.path.exists(lm_preprocessed_dir):
        os.makedirs(lm_preprocessed_dir)
    if not os.path.exists(backwards_preprocessed_dir):
        os.makedirs(backwards_preprocessed_dir)

    score1_file = rerank_utils.rescore_file_name(pre_gen, args.prefix_len, args.model1_name,
                                                 target_prefix_frac=args.target_prefix_frac,
                                                 source_prefix_frac=args.source_prefix_frac,
                                                 backwards=args.backwards1)
    if args.score_model2 is not None:
        score2_file = rerank_utils.rescore_file_name(pre_gen, args.prefix_len, args.model2_name,
                                                     target_prefix_frac=args.target_prefix_frac,
                                                     source_prefix_frac=args.source_prefix_frac,
                                                     backwards=args.backwards2)

    predictions_bpe_file = pre_gen+"/generate_output_bpe.txt"

    using_nbest = args.nbest_list is not None

    if using_nbest:
        print("Using predefined n-best list from interactive.py")
        predictions_bpe_file = args.nbest_list

    else:
        if not os.path.isfile(predictions_bpe_file):
            print("STEP 1: generate predictions using the p(T|S) model with bpe")
            print(args.data)
            param1 = [args.data,
                      "--path", args.gen_model,
                      "--shard-id", str(args.shard_id),
                      "--num-shards", str(args.num_shards),
                      "--nbest", str(args.num_rescore),
                      "--batch-size", str(args.batch_size),
                      "--beam", str(args.num_rescore),
                      "--max-sentences", str(args.num_rescore),
                      "--gen-subset", args.gen_subset,
                      "--source-lang", args.source_lang,
                      "--target-lang", args.target_lang]
            if args.sampling:
                param1 += ["--sampling"]

            gen_parser = options.get_generation_parser()
            input_args = options.parse_args_and_arch(gen_parser, param1)

            print(input_args)
            with open(predictions_bpe_file, 'w') as f:
                with redirect_stdout(f):
                    generate.main(input_args)

    gen_output = rerank_utils.BitextOutputFromGen(predictions_bpe_file, bpe_symbol=args.remove_bpe,
                                                  nbest=using_nbest, prefix_len=args.prefix_len,
                                                  target_prefix_frac=args.target_prefix_frac)

    if args.diff_bpe:
        rerank_utils.write_reprocessed(gen_output.no_bpe_source, gen_output.no_bpe_hypo,
                                       gen_output.no_bpe_target, pre_gen+"/source_gen_bpe."+args.source_lang,
                                       pre_gen+"/target_gen_bpe."+args.target_lang,
                                       pre_gen+"/reference_gen_bpe."+args.target_lang)
        bitext_bpe = args.rescore_bpe_code
        bpe_src_param = ["-c", bitext_bpe,
                         "--input", pre_gen+"/source_gen_bpe."+args.source_lang,
                         "--output", pre_gen+"/rescore_data."+args.source_lang]
        bpe_tgt_param = ["-c", bitext_bpe,
                         "--input", pre_gen+"/target_gen_bpe."+args.target_lang,
                         "--output", pre_gen+"/rescore_data."+args.target_lang]

        subprocess.call(["python",
                         os.path.join(os.path.dirname(__file__),
                                      "subword-nmt/subword_nmt/apply_bpe.py")] + bpe_src_param,
                        shell=False)

        subprocess.call(["python",
                         os.path.join(os.path.dirname(__file__),
                                      "subword-nmt/subword_nmt/apply_bpe.py")] + bpe_tgt_param,
                        shell=False)

    if (not os.path.isfile(score1_file) and not rerank1_is_gen) or \
            (args.score_model2 is not None and not os.path.isfile(score2_file) and not rerank2_is_gen):
        print("STEP 2: process the output of generate.py so we have clean text files with the translations")

        rescore_file = "/rescore_data"
        if args.prefix_len is not None:
            prefix_len_rescore_file = rescore_file + "prefix"+str(args.prefix_len)
        if args.target_prefix_frac is not None:
            target_prefix_frac_rescore_file = rescore_file + "target_prefix_frac"+str(args.target_prefix_frac)
        if args.source_prefix_frac is not None:
            source_prefix_frac_rescore_file = rescore_file + "source_prefix_frac"+str(args.source_prefix_frac)

        if not args.right_to_left1 or not args.right_to_left2:
            if not args.diff_bpe:
                rerank_utils.write_reprocessed(gen_output.source, gen_output.hypo, gen_output.target,
                                               pre_gen+rescore_file+"."+args.source_lang,
                                               pre_gen+rescore_file+"."+args.target_lang,
                                               pre_gen+"/reference_file", bpe_symbol=args.remove_bpe)
                if args.prefix_len is not None:
                    bw_rescore_file = prefix_len_rescore_file
                    rerank_utils.write_reprocessed(gen_output.source, gen_output.hypo, gen_output.target,
                                                   pre_gen+prefix_len_rescore_file+"."+args.source_lang,
                                                   pre_gen+prefix_len_rescore_file+"."+args.target_lang,
                                                   pre_gen+"/reference_file", prefix_len=args.prefix_len,
                                                   bpe_symbol=args.remove_bpe)
                elif args.target_prefix_frac is not None:
                    bw_rescore_file = target_prefix_frac_rescore_file
                    rerank_utils.write_reprocessed(gen_output.source, gen_output.hypo, gen_output.target,
                                                   pre_gen+target_prefix_frac_rescore_file+"."+args.source_lang,
                                                   pre_gen+target_prefix_frac_rescore_file+"."+args.target_lang,
                                                   pre_gen+"/reference_file", bpe_symbol=args.remove_bpe,
                                                   target_prefix_frac=args.target_prefix_frac)
                else:
                    bw_rescore_file = rescore_file

                if args.source_prefix_frac is not None:
                    fw_rescore_file = source_prefix_frac_rescore_file
                    rerank_utils.write_reprocessed(gen_output.source, gen_output.hypo, gen_output.target,
                                                   pre_gen+source_prefix_frac_rescore_file+"."+args.source_lang,
                                                   pre_gen+source_prefix_frac_rescore_file+"."+args.target_lang,
                                                   pre_gen+"/reference_file", bpe_symbol=args.remove_bpe,
                                                   source_prefix_frac=args.source_prefix_frac)
                else:
                    fw_rescore_file = rescore_file

        if args.right_to_left1 or args.right_to_left2:
            rerank_utils.write_reprocessed(gen_output.source, gen_output.hypo, gen_output.target,
                                           pre_gen+"/right_to_left_rescore_data."+args.source_lang,
                                           pre_gen+"/right_to_left_rescore_data."+args.target_lang,
                                           pre_gen+"/right_to_left_reference_file",
                                           right_to_left=True, bpe_symbol=args.remove_bpe)

        print("STEP 3: binarize the translations")
        if not args.right_to_left1 or args.score_model2 is not None and not args.right_to_left2 or not rerank1_is_gen:

            if args.backwards1 or args.backwards2:
                if args.backwards_score_dict_dir is not None:
                    bw_dict = args.backwards_score_dict_dir
                else:
                    bw_dict = args.score_dict_dir
                bw_preprocess_param = ["--source-lang", scorer1_src,
                                       "--target-lang", scorer1_tgt,
                                       "--trainpref", pre_gen+bw_rescore_file,
                                       "--srcdict", bw_dict + "/dict." + scorer1_src + ".txt",
                                       "--tgtdict", bw_dict + "/dict." + scorer1_tgt + ".txt",
                                       "--destdir", backwards_preprocessed_dir]
                preprocess_parser = options.get_preprocessing_parser()
                input_args = preprocess_parser.parse_args(bw_preprocess_param)
                preprocess.main(input_args)

            preprocess_param = ["--source-lang", scorer1_src,
                                "--target-lang", scorer1_tgt,
                                "--trainpref", pre_gen+fw_rescore_file,
                                "--srcdict", args.score_dict_dir+"/dict."+scorer1_src+".txt",
                                "--tgtdict", args.score_dict_dir+"/dict."+scorer1_tgt+".txt",
                                "--destdir", left_to_right_preprocessed_dir]
            preprocess_parser = options.get_preprocessing_parser()
            input_args = preprocess_parser.parse_args(preprocess_param)
            preprocess.main(input_args)

        if args.right_to_left1 or args.right_to_left2:
            preprocess_param = ["--source-lang", scorer1_src,
                                "--target-lang", scorer1_tgt,
                                "--trainpref", pre_gen+"/right_to_left_rescore_data",
                                "--srcdict", args.score_dict_dir+"/dict."+scorer1_src+".txt",
                                "--tgtdict", args.score_dict_dir+"/dict."+scorer1_tgt+".txt",
                                "--destdir", right_to_left_preprocessed_dir]
            preprocess_parser = options.get_preprocessing_parser()
            input_args = preprocess_parser.parse_args(preprocess_param)
            preprocess.main(input_args)

    return gen_output


def cli_main():
    parser = rerank_options.get_reranking_parser()
    args = options.parse_args_and_arch(parser)
    gen_and_reprocess_nbest(args)


if __name__ == '__main__':
    cli_main()
