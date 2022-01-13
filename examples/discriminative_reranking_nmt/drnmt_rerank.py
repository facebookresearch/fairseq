#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Score raw text with a trained model.
"""

from collections import namedtuple
import logging
from multiprocessing import Pool
import sys
import os
import random

import numpy as np
import sacrebleu
import torch

from fairseq import checkpoint_utils, options, utils


logger = logging.getLogger("fairseq_cli.drnmt_rerank")
logger.setLevel(logging.INFO)

Batch = namedtuple("Batch", "ids src_tokens src_lengths")


pool_init_variables = {}


def init_loaded_scores(mt_scores, model_scores, hyp, ref):
    global pool_init_variables
    pool_init_variables["mt_scores"] = mt_scores
    pool_init_variables["model_scores"] = model_scores
    pool_init_variables["hyp"] = hyp
    pool_init_variables["ref"] = ref


def parse_fairseq_gen(filename, task):
    source = {}
    hypos = {}
    scores = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("S-"):  # source
                uid, text = line.split("\t", 1)
                uid = int(uid[2:])
                source[uid] = text
            elif line.startswith("D-"):  # hypo
                uid, score, text = line.split("\t", 2)
                uid = int(uid[2:])
                if uid not in hypos:
                    hypos[uid] = []
                    scores[uid] = []
                hypos[uid].append(text)
                scores[uid].append(float(score))
            else:
                continue

    source_out = [source[i] for i in range(len(hypos))]
    hypos_out = [h for i in range(len(hypos)) for h in hypos[i]]
    scores_out = [s for i in range(len(scores)) for s in scores[i]]

    return source_out, hypos_out, scores_out


def read_target(filename):
    with open(filename, "r", encoding="utf-8") as f:
        output = [line.strip() for line in f]
    return output


def make_batches(args, src, hyp, task, max_positions, encode_fn):
    assert len(src) * args.beam == len(
        hyp
    ), f"Expect {len(src) * args.beam} hypotheses for {len(src)} source sentences with beam size {args.beam}. Got {len(hyp)} hypotheses intead."
    hyp_encode = [
        task.source_dictionary.encode_line(encode_fn(h), add_if_not_exist=False).long()
        for h in hyp
    ]
    if task.cfg.include_src:
        src_encode = [
            task.source_dictionary.encode_line(
                encode_fn(s), add_if_not_exist=False
            ).long()
            for s in src
        ]
        tokens = [(src_encode[i // args.beam], h) for i, h in enumerate(hyp_encode)]
        lengths = [(t1.numel(), t2.numel()) for t1, t2 in tokens]
    else:
        tokens = [(h,) for h in hyp_encode]
        lengths = [(h.numel(),) for h in hyp_encode]

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)

    for batch in itr:
        yield Batch(
            ids=batch["id"],
            src_tokens=batch["net_input"]["src_tokens"],
            src_lengths=batch["net_input"]["src_lengths"],
        )


def decode_rerank_scores(args):
    if args.max_tokens is None and args.batch_size is None:
        args.batch_size = 1

    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path], arg_overrides=eval(args.model_overrides),
    )

    for model in models:
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(args)
    bpe = task.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    src, hyp, mt_scores = parse_fairseq_gen(args.in_text, task)
    model_scores = {}
    logger.info("decode reranker score")
    for batch in make_batches(args, src, hyp, task, max_positions, encode_fn):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()

        sample = {
            "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths},
        }
        scores = task.inference_step(generator, models, sample)

        for id, sc in zip(batch.ids.tolist(), scores.tolist()):
            model_scores[id] = sc[0]

    model_scores = [model_scores[i] for i in range(len(model_scores))]

    return src, hyp, mt_scores, model_scores


def get_score(mt_s, md_s, w1, lp, tgt_len):
    return mt_s / (tgt_len ** lp) * w1 + md_s


def get_best_hyps(mt_scores, md_scores, hypos, fw_weight, lenpen, beam):
    assert len(mt_scores) == len(md_scores) and len(mt_scores) == len(hypos)
    hypo_scores = []
    best_hypos = []
    best_scores = []
    offset = 0
    for i in range(len(hypos)):
        tgt_len = len(hypos[i].split())
        hypo_scores.append(
            get_score(mt_scores[i], md_scores[i], fw_weight, lenpen, tgt_len)
        )

        if (i + 1) % beam == 0:
            max_i = np.argmax(hypo_scores)
            best_hypos.append(hypos[offset + max_i])
            best_scores.append(hypo_scores[max_i])
            hypo_scores = []
            offset += beam
    return best_hypos, best_scores


def eval_metric(args, hypos, ref):
    if args.metric == "bleu":
        score = sacrebleu.corpus_bleu(hypos, [ref]).score
    else:
        score = sacrebleu.corpus_ter(hypos, [ref]).score

    return score


def score_target_hypo(args, fw_weight, lp):
    mt_scores = pool_init_variables["mt_scores"]
    model_scores = pool_init_variables["model_scores"]
    hyp = pool_init_variables["hyp"]
    ref = pool_init_variables["ref"]
    best_hypos, _ = get_best_hyps(
        mt_scores, model_scores, hyp, fw_weight, lp, args.beam
    )
    rerank_eval = None
    if ref:
        rerank_eval = eval_metric(args, best_hypos, ref)
        print(f"fw_weight {fw_weight}, lenpen {lp}, eval {rerank_eval}")

    return rerank_eval


def print_result(best_scores, best_hypos, output_file):
    for i, (s, h) in enumerate(zip(best_scores, best_hypos)):
        print(f"{i}\t{s}\t{h}", file=output_file)


def main(args):
    utils.import_user_module(args)

    src, hyp, mt_scores, model_scores = decode_rerank_scores(args)

    assert (
        not args.tune or args.target_text is not None
    ), "--target-text has to be set when tuning weights"
    if args.target_text:
        ref = read_target(args.target_text)
        assert len(src) == len(
            ref
        ), f"different numbers of source and target sentences ({len(src)} vs. {len(ref)})"

        orig_best_hypos = [hyp[i] for i in range(0, len(hyp), args.beam)]
        orig_eval = eval_metric(args, orig_best_hypos, ref)

    if args.tune:
        logger.info("tune weights for reranking")

        random_params = np.array(
            [
                [
                    random.uniform(
                        args.lower_bound_fw_weight, args.upper_bound_fw_weight
                    ),
                    random.uniform(args.lower_bound_lenpen, args.upper_bound_lenpen),
                ]
                for k in range(args.num_trials)
            ]
        )

        logger.info("launching pool")
        with Pool(
            32,
            initializer=init_loaded_scores,
            initargs=(mt_scores, model_scores, hyp, ref),
        ) as p:
            rerank_scores = p.starmap(
                score_target_hypo,
                [
                    (args, random_params[i][0], random_params[i][1],)
                    for i in range(args.num_trials)
                ],
            )
        if args.metric == "bleu":
            best_index = np.argmax(rerank_scores)
        else:
            best_index = np.argmin(rerank_scores)
        best_fw_weight = random_params[best_index][0]
        best_lenpen = random_params[best_index][1]
    else:
        assert (
            args.lenpen is not None and args.fw_weight is not None
        ), "--lenpen and --fw-weight should be set"
        best_fw_weight, best_lenpen = args.fw_weight, args.lenpen

    best_hypos, best_scores = get_best_hyps(
        mt_scores, model_scores, hyp, best_fw_weight, best_lenpen, args.beam
    )

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(
            args.results_path, "generate-{}.txt".format(args.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as o:
            print_result(best_scores, best_hypos, o)
    else:
        print_result(best_scores, best_hypos, sys.stdout)

    if args.target_text:
        rerank_eval = eval_metric(args, best_hypos, ref)
        print(f"before reranking, {args.metric.upper()}:", orig_eval)
        print(
            f"after reranking with fw_weight={best_fw_weight}, lenpen={best_lenpen}, {args.metric.upper()}:",
            rerank_eval,
        )


def cli_main():
    parser = options.get_generation_parser(interactive=True)

    parser.add_argument(
        "--in-text",
        default=None,
        required=True,
        help="text from fairseq-interactive output, containing source sentences and hypotheses",
    )
    parser.add_argument("--target-text", default=None, help="reference text")
    parser.add_argument("--metric", type=str, choices=["bleu", "ter"], default="bleu")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="if set, tune weights on fw scores and lenpen instead of applying fixed weights for reranking",
    )
    parser.add_argument(
        "--lower-bound-fw-weight",
        default=0.0,
        type=float,
        help="lower bound of search space",
    )
    parser.add_argument(
        "--upper-bound-fw-weight",
        default=3,
        type=float,
        help="upper bound of search space",
    )
    parser.add_argument(
        "--lower-bound-lenpen",
        default=0.0,
        type=float,
        help="lower bound of search space",
    )
    parser.add_argument(
        "--upper-bound-lenpen",
        default=3,
        type=float,
        help="upper bound of search space",
    )
    parser.add_argument(
        "--fw-weight", type=float, default=None, help="weight on the fw model score"
    )
    parser.add_argument(
        "--num-trials",
        default=1000,
        type=int,
        help="number of trials to do for random search",
    )

    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
