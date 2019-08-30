#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import logging
import os

import sentencepiece as spm
import torch
from fairseq import options, progress_bar, utils, tasks
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.utils import import_user_module


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_asr_eval_argument(parser):
    parser.add_argument("--ctc", action="store_true", help="decode a ctc model")
    parser.add_argument("--rnnt", default=False, help="decode a rnnt model")
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary\
output units",
    )
    parser.add_argument(
        "--lm_weight",
        default=0.2,
        help="weight for wfstlm while interpolating\
with neural score",
    )
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    return parser


def check_args(args):
    assert args.path is not None, "--path required for generation!"
    assert args.results_path is not None, "--results_path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def get_dataset_itr(args, task):
    return task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=(1000000.0, 1000000.0),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)


def process_predictions(args, hypos, sp, tgt_dict, target_tokens, res_files, speaker, id):
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())
        hyp_words = sp.DecodePieces(hyp_pieces.split())
        print(
            "{} ({}-{})".format(hyp_pieces, speaker, id),
            file=res_files["hypo.units"],
        )
        print(
            "{} ({}-{})".format(hyp_words, speaker, id),
            file=res_files["hypo.words"],
        )

        tgt_pieces = tgt_dict.string(target_tokens)
        tgt_words = sp.DecodePieces(tgt_pieces.split())
        print(
            "{} ({}-{})".format(tgt_pieces, speaker, id),
            file=res_files["ref.units"],
        )
        print(
            "{} ({}-{})".format(tgt_words, speaker, id),
            file=res_files["ref.words"],
        )
        # only score top hypothesis
        if not args.quiet:
            logger.debug("HYPO:" + hyp_words)
            logger.debug("TARGET:" + tgt_words)
            logger.debug("___________________")


def prepare_result_files(args):
    def get_res_file(file_prefix):
        path = os.path.join(
            args.results_path,
            "{}-{}-{}.txt".format(
                file_prefix, os.path.basename(args.path), args.gen_subset
            ),
        )
        return open(path, "w", buffering=1)

    return {
        "hypo.words": get_res_file("hypo.word"),
        "hypo.units": get_res_file("hypo.units"),
        "ref.words": get_res_file("ref.word"),
        "ref.units": get_res_file("ref.units"),
    }


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation
    """
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


def main(args):
    check_args(args)
    import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 30000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    logger.info(
        "| {} {} {} examples".format(
            args.data, args.gen_subset, len(task.dataset(args.gen_subset))
        )
    )

    # Set dictionary
    tgt_dict = task.target_dictionary

    if args.ctc or args.rnnt:
        tgt_dict.add_symbol("<ctc_blank>")
        if args.ctc:
            logger.info("| decoding a ctc model")
        if args.rnnt:
            logger.info("| decoding a rnnt model")

    # Load ensemble
    logger.info("| loading model(s) from {}".format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(":"),
        task,
        model_arg_overrides=eval(args.model_overrides),  # noqa
    )
    optimize_models(args, use_cuda, models)

    # Load dataset (possibly sharded)
    itr = get_dataset_itr(args, task)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    num_sentences = 0

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(args.data, 'spm.model'))

    res_files = prepare_result_files(args)
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample["target"][:, : args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                speaker = task.dataset(args.gen_subset).speakers[int(sample_id)]
                id = task.dataset(args.gen_subset).ids[int(sample_id)]
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )
                # Process top predictions
                process_predictions(
                    args, hypos[i], sp, tgt_dict, target_tokens, res_files, speaker, id
                )

            wps_meter.update(num_generated_tokens)
            t.log({"wps": round(wps_meter.avg)})
            num_sentences += sample["nsentences"]

    logger.info(
        "| Processed {} sentences ({} tokens) in {:.1f}s ({:.2f}"
        "sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    logger.info("| Generate {} with beam={}".format(args.gen_subset, args.beam))


def cli_main():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
