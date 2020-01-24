#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import logging
import math
import os

import sentencepiece as spm
import torch
from fairseq import checkpoint_utils, options, progress_bar, utils, tasks
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.utils import import_user_module


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_asr_eval_argument(parser):
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
        "--lm-weight",
        "--lm_weight",
        type=float,
        default=0.2,
        help="weight for lm while interpolating with neural score",
    )
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    parser.add_argument(
        "--w2l-decoder", choices=["viterbi", "kenlm"], help="use a w2l decoder"
    )
    parser.add_argument("--lexicon", help="lexicon for w2l decoder")
    parser.add_argument("--kenlm-model", help="kenlm model for w2l decoder")
    parser.add_argument("--beam-threshold", type=float, default=25.0)
    parser.add_argument("--word-score", type=float, default=1.0)
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    return parser


def check_args(args):
    assert args.path is not None, "--path required for generation!"
    assert args.results_path is not None, "--results_path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"


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


def process_predictions(
    args, hypos, sp, tgt_dict, target_tokens, res_files, speaker, id
):
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())
        hyp_words = sp.DecodePieces(hyp_pieces.split())
        print(
            "{} ({}-{})".format(hyp_pieces, speaker, id), file=res_files["hypo.units"]
        )
        print("{} ({}-{})".format(hyp_words, speaker, id), file=res_files["hypo.words"])

        tgt_pieces = tgt_dict.string(target_tokens)
        tgt_words = sp.DecodePieces(tgt_pieces.split())
        print("{} ({}-{})".format(tgt_pieces, speaker, id), file=res_files["ref.units"])
        print("{} ({}-{})".format(tgt_words, speaker, id), file=res_files["ref.words"])
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


def load_models_and_criterions(filenames, arg_overrides=None, task=None):
    models = []
    criterions = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))
        state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)

        args = state["args"]
        if task is None:
            task = tasks.setup_task(args)

        # build model for ensemble
        model = task.build_model(args)
        model.load_state_dict(state["model"], strict=True)
        models.append(model)

        criterion = task.build_criterion(args)
        if "criterion" in state:
            criterion.load_state_dict(state["criterion"], strict=True)
        criterions.append(criterion)
    return models, criterions, args


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

    logger.info("| decoding with criterion {}".format(args.criterion))

    # Load ensemble
    logger.info("| loading model(s) from {}".format(args.path))
    models, criterions, _model_args = load_models_and_criterions(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),  # noqa
        task=task,
    )
    optimize_models(args, use_cuda, models)

    # hack to pass transitions to W2lDecoder
    if args.criterion == "asg_loss":
        trans = criterions[0].asg.trans.data
        args.asg_transitions = torch.flatten(trans).tolist()

    # Load dataset (possibly sharded)
    itr = get_dataset_itr(args, task)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    num_sentences = 0

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(args.data, "spm.model"))

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

            for i, sample_id in enumerate(sample["id"].tolist()):
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
