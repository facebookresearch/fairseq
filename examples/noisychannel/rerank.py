import rerank_utils
import rerank_generate
import rerank_score_bw
import rerank_score_lm
from fairseq import bleu, options
from fairseq.data import dictionary
from examples.noisychannel import rerank_options
from multiprocessing import Pool

import math
import numpy as np


def score_target_hypo(args, a, b, c, lenpen, target_outfile, hypo_outfile, write_hypos, normalize):

    print("lenpen", lenpen, "weight1", a, "weight2", b, "weight3", c)
    gen_output_lst, bitext1_lst, bitext2_lst, lm_res_lst = load_score_files(args)
    dict = dictionary.Dictionary()
    scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())

    ordered_hypos = {}
    ordered_targets = {}

    for shard_id in range(len(bitext1_lst)):
        bitext1 = bitext1_lst[shard_id]
        bitext2 = bitext2_lst[shard_id]
        gen_output = gen_output_lst[shard_id]
        lm_res = lm_res_lst[shard_id]

        total = len(bitext1.rescore_source.keys())
        source_lst = []
        hypo_lst = []
        score_lst = []
        reference_lst = []
        j = 1
        best_score = -math.inf

        for i in range(total):
            # length is measured in terms of words, not bpe tokens, since models may not share the same bpe
            target_len = len(bitext1.rescore_hypo[i].split())

            if lm_res is not None:
                lm_score = lm_res.score[i]
            else:
                lm_score = 0

            if bitext2 is not None:
                bitext2_score = bitext2.rescore_score[i]
                bitext2_backwards = bitext2.backwards
            else:
                bitext2_score = None
                bitext2_backwards = None

            score = rerank_utils.get_score(a, b, c, target_len,
                                           bitext1.rescore_score[i], bitext2_score, lm_score=lm_score,
                                           lenpen=lenpen, src_len=bitext1.source_lengths[i],
                                           tgt_len=bitext1.target_lengths[i], bitext1_backwards=bitext1.backwards,
                                           bitext2_backwards=bitext2_backwards, normalize=normalize)

            if score > best_score:
                best_score = score
                best_hypo = bitext1.rescore_hypo[i]

            if j == gen_output.num_hypos[i] or j == args.num_rescore:
                j = 1
                hypo_lst.append(best_hypo)
                score_lst.append(best_score)
                source_lst.append(bitext1.rescore_source[i])
                reference_lst.append(bitext1.rescore_target[i])

                best_score = -math.inf
                best_hypo = ""
            else:
                j += 1

        gen_keys = list(sorted(gen_output.no_bpe_target.keys()))

        for key in range(len(gen_keys)):
            if args.prefix_len is None:
                assert hypo_lst[key] in gen_output.no_bpe_hypo[gen_keys[key]], (
                    "pred and rescore hypo mismatch: i: " + str(key) + ", "
                    + str(hypo_lst[key]) + str(gen_keys[key])
                    + str(gen_output.no_bpe_hypo[key])
                )
                sys_tok = dict.encode_line(hypo_lst[key])
                ref_tok = dict.encode_line(gen_output.no_bpe_target[gen_keys[key]])
                scorer.add(ref_tok, sys_tok)

            else:
                full_hypo = rerank_utils.get_full_from_prefix(hypo_lst[key], gen_output.no_bpe_hypo[gen_keys[key]])
                sys_tok = dict.encode_line(full_hypo)
                ref_tok = dict.encode_line(gen_output.no_bpe_target[gen_keys[key]])
                scorer.add(ref_tok, sys_tok)

        # if only one set of hyper parameters is provided, write the predictions to a file
        if write_hypos:
            # recover the orinal ids from n best list generation
            for key in range(len(gen_output.no_bpe_target)):
                if args.prefix_len is None:
                    assert hypo_lst[key] in gen_output.no_bpe_hypo[gen_keys[key]], \
                        "pred and rescore hypo mismatch:"+"i:"+str(key)+str(hypo_lst[key]) + str(gen_output.no_bpe_hypo[key])
                    ordered_hypos[gen_keys[key]] = hypo_lst[key]
                    ordered_targets[gen_keys[key]] = gen_output.no_bpe_target[gen_keys[key]]

                else:
                    full_hypo = rerank_utils.get_full_from_prefix(hypo_lst[key], gen_output.no_bpe_hypo[gen_keys[key]])
                    ordered_hypos[gen_keys[key]] = full_hypo
                    ordered_targets[gen_keys[key]] = gen_output.no_bpe_target[gen_keys[key]]

    # write the hypos in the original order from nbest list generation
    if args.num_shards == (len(bitext1_lst)):
        with open(target_outfile, 'w') as t:
            with open(hypo_outfile, 'w') as h:
                for key in range(len(ordered_hypos)):
                    t.write(ordered_targets[key])
                    h.write(ordered_hypos[key])

    res = scorer.result_string(4)
    if write_hypos:
        print(res)
    score = rerank_utils.parse_bleu_scoring(res)
    return score


def match_target_hypo(args, target_outfile, hypo_outfile):
    """combine scores from the LM and bitext models, and write the top scoring hypothesis to a file"""
    if len(args.weight1) == 1:
        res = score_target_hypo(args, args.weight1[0], args.weight2[0],
                                args.weight3[0], args.lenpen[0], target_outfile,
                                hypo_outfile, True, args.normalize)
        rerank_scores = [res]
    else:
        print("launching pool")
        with Pool(32) as p:
            rerank_scores = p.starmap(score_target_hypo,
                                      [(args, args.weight1[i], args.weight2[i], args.weight3[i],
                                        args.lenpen[i], target_outfile, hypo_outfile,
                                        False, args.normalize) for i in range(len(args.weight1))])

    if len(rerank_scores) > 1:
        best_index = np.argmax(rerank_scores)
        best_score = rerank_scores[best_index]
        print("best score", best_score)
        print("best lenpen", args.lenpen[best_index])
        print("best weight1", args.weight1[best_index])
        print("best weight2", args.weight2[best_index])
        print("best weight3", args.weight3[best_index])
        return args.lenpen[best_index], args.weight1[best_index], \
            args.weight2[best_index], args.weight3[best_index], best_score

    else:
        return args.lenpen[0], args.weight1[0], args.weight2[0], args.weight3[0], rerank_scores[0]


def load_score_files(args):
    if args.all_shards:
        shard_ids = list(range(args.num_shards))
    else:
        shard_ids = [args.shard_id]

    gen_output_lst = []
    bitext1_lst = []
    bitext2_lst = []
    lm_res1_lst = []

    for shard_id in shard_ids:
        using_nbest = args.nbest_list is not None
        pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, \
            backwards_preprocessed_dir, lm_preprocessed_dir = \
            rerank_utils.get_directories(args.data_dir_name, args.num_rescore, args.gen_subset,
                                         args.gen_model_name, shard_id, args.num_shards, args.sampling,
                                         args.prefix_len, args.target_prefix_frac, args.source_prefix_frac)

        rerank1_is_gen = args.gen_model == args.score_model1 and args.source_prefix_frac is None
        rerank2_is_gen = args.gen_model == args.score_model2 and args.source_prefix_frac is None

        score1_file = rerank_utils.rescore_file_name(pre_gen, args.prefix_len, args.model1_name,
                                                     target_prefix_frac=args.target_prefix_frac,
                                                     source_prefix_frac=args.source_prefix_frac,
                                                     backwards=args.backwards1)
        if args.score_model2 is not None:
            score2_file = rerank_utils.rescore_file_name(pre_gen, args.prefix_len, args.model2_name,
                                                         target_prefix_frac=args.target_prefix_frac,
                                                         source_prefix_frac=args.source_prefix_frac,
                                                         backwards=args.backwards2)
        if args.language_model is not None:
            lm_score_file = rerank_utils.rescore_file_name(pre_gen, args.prefix_len, args.lm_name, lm_file=True)

        # get gen output
        predictions_bpe_file = pre_gen+"/generate_output_bpe.txt"
        if using_nbest:
            print("Using predefined n-best list from interactive.py")
            predictions_bpe_file = args.nbest_list
        gen_output = rerank_utils.BitextOutputFromGen(predictions_bpe_file, bpe_symbol=args.remove_bpe,
                                                      nbest=using_nbest, prefix_len=args.prefix_len,
                                                      target_prefix_frac=args.target_prefix_frac)

        if rerank1_is_gen:
            bitext1 = gen_output
        else:
            bitext1 = rerank_utils.BitextOutput(score1_file, args.backwards1, args.right_to_left1,
                                                args.remove_bpe, args.prefix_len, args.target_prefix_frac,
                                                args.source_prefix_frac)

        if args.score_model2 is not None or args.nbest_list is not None:
            if rerank2_is_gen:
                bitext2 = gen_output
            else:
                bitext2 = rerank_utils.BitextOutput(score2_file, args.backwards2, args.right_to_left2,
                                                    args.remove_bpe, args.prefix_len, args.target_prefix_frac,
                                                    args.source_prefix_frac)

                assert bitext2.source_lengths == bitext1.source_lengths, \
                    "source lengths for rescoring models do not match"
                assert bitext2.target_lengths == bitext1.target_lengths, \
                    "target lengths for rescoring models do not match"
        else:
            if args.diff_bpe:
                assert args.score_model2 is None
                bitext2 = gen_output
            else:
                bitext2 = None

        if args.language_model is not None:
            lm_res1 = rerank_utils.LMOutput(lm_score_file, args.lm_dict, args.prefix_len,
                                            args.remove_bpe, args.target_prefix_frac)
        else:
            lm_res1 = None

        gen_output_lst.append(gen_output)
        bitext1_lst.append(bitext1)
        bitext2_lst.append(bitext2)
        lm_res1_lst.append(lm_res1)
    return gen_output_lst, bitext1_lst, bitext2_lst, lm_res1_lst


def rerank(args):
    if type(args.lenpen) is not list:
        args.lenpen = [args.lenpen]
    if type(args.weight1) is not list:
        args.weight1 = [args.weight1]
    if type(args.weight2) is not list:
        args.weight2 = [args.weight2]
    if type(args.weight3) is not list:
        args.weight3 = [args.weight3]
    if args.all_shards:
        shard_ids = list(range(args.num_shards))
    else:
        shard_ids = [args.shard_id]

    for shard_id in shard_ids:
        pre_gen, left_to_right_preprocessed_dir, right_to_left_preprocessed_dir, \
                backwards_preprocessed_dir, lm_preprocessed_dir = \
                rerank_utils.get_directories(args.data_dir_name, args.num_rescore, args.gen_subset,
                                             args.gen_model_name, shard_id, args.num_shards, args.sampling,
                                             args.prefix_len, args.target_prefix_frac, args.source_prefix_frac)
        rerank_generate.gen_and_reprocess_nbest(args)
        rerank_score_bw.score_bw(args)
        rerank_score_lm.score_lm(args)

        if args.write_hypos is None:
            write_targets = pre_gen+"/matched_targets"
            write_hypos = pre_gen+"/matched_hypos"
        else:
            write_targets = args.write_hypos+"_targets" + args.gen_subset
            write_hypos = args.write_hypos+"_hypos" + args.gen_subset

    if args.all_shards:
        write_targets += "_all_shards"
        write_hypos += "_all_shards"

    best_lenpen, best_weight1, best_weight2, best_weight3, best_score = \
        match_target_hypo(args, write_targets, write_hypos)

    return best_lenpen, best_weight1, best_weight2, best_weight3, best_score


def cli_main():
    parser = rerank_options.get_reranking_parser()
    args = options.parse_args_and_arch(parser)
    rerank(args)


if __name__ == '__main__':
    cli_main()
