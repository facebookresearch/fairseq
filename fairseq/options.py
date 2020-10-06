# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Callable, List, Optional

import torch
from fairseq import utils
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass.data_class import (
    CheckpointParams,
    CommonEvalParams,
    CommonParams,
    DatasetParams,
    DistributedTrainingParams,
    EvalLMParams,
    OptimizationParams,
)
from fairseq.dataclass.utils import gen_parser_from_dataclass

# this import is for backward compatibility
from fairseq.utils import csv_str_list, eval_bool, eval_str_dict, eval_str_list  # noqa


def get_preprocessing_parser(default_task="translation"):
    parser = get_parser("Preprocessing", default_task)
    add_preprocess_args(parser)
    return parser


def get_training_parser(default_task="translation"):
    parser = get_parser("Trainer", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser


def get_generation_parser(interactive=False, default_task="translation"):
    parser = get_parser("Generation", default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_generation_args(parser)
    if interactive:
        add_interactive_args(parser)
    return parser


def get_interactive_generation_parser(default_task="translation"):
    return get_generation_parser(interactive=True, default_task=default_task)


def get_eval_lm_parser(default_task="language_modeling"):
    parser = get_parser("Evaluate Language Model", default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_eval_lm_args(parser)
    return parser


def get_validation_parser(default_task=None):
    parser = get_parser("Validation", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser, default_world_size=1)
    group = parser.add_argument_group("Evaluation")
    gen_parser_from_dataclass(group, CommonEvalParams())
    return parser


def parse_args_and_arch(
    parser: argparse.ArgumentParser,
    input_args: List[str] = None,
    parse_known: bool = False,
    suppress_defaults: bool = False,
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    """
    Args:
        parser (ArgumentParser): the parser
        input_args (List[str]): strings to parse, defaults to sys.argv
        parse_known (bool): only parse known arguments, similar to
            `ArgumentParser.parse_known_args`
        suppress_defaults (bool): parse while ignoring all default values
        modify_parser (Optional[Callable[[ArgumentParser], None]]):
            function to modify the parser, e.g., to set default values
    """
    if suppress_defaults:
        # Parse args without any default values. This requires us to parse
        # twice, once to identify all the necessary task/model args, and a second
        # time with all defaults set to None.
        args = parse_args_and_arch(
            parser,
            input_args=input_args,
            parse_known=parse_known,
            suppress_defaults=False,
        )
        suppressed_parser = argparse.ArgumentParser(add_help=False, parents=[parser])
        suppressed_parser.set_defaults(**{k: None for k, v in vars(args).items()})
        args = suppressed_parser.parse_args(input_args)
        return argparse.Namespace(
            **{k: v for k, v in vars(args).items() if v is not None}
        )

    from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY, MODEL_REGISTRY

    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args(input_args)
    utils.import_user_module(usr_args)

    if modify_parser is not None:
        modify_parser(parser)

    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, "arch"):
        model_specific_group = parser.add_argument_group(
            "Model-specific configuration",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        if args.arch in ARCH_MODEL_REGISTRY:
            ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        elif args.arch in MODEL_REGISTRY:
            MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        else:
            raise RuntimeError()

    # Add *-specific args to parser.
    from fairseq.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)
    if hasattr(args, "task"):
        from fairseq.tasks import TASK_REGISTRY

        TASK_REGISTRY[args.task].add_args(parser)
    if getattr(args, "use_bmuf", False):
        # hack to support extra args for block distributed data parallelism
        from fairseq.optim.bmuf import FairseqBMUF

        FairseqBMUF.add_args(parser)

    # Modify the parser a second time, since defaults may have been reset
    if modify_parser is not None:
        modify_parser(parser)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None
    # Post-process args.
    if hasattr(args, "batch_size_valid") and args.batch_size_valid is None:
        args.batch_size_valid = args.batch_size
    if hasattr(args, "max_tokens_valid") and args.max_tokens_valid is None:
        args.max_tokens_valid = args.max_tokens
    if getattr(args, "memory_efficient_fp16", False):
        args.fp16 = True
    if getattr(args, "memory_efficient_bf16", False):
        args.bf16 = True
    args.tpu = getattr(args, "tpu", False)
    args.bf16 = getattr(args, "bf16", False)
    if args.bf16:
        args.tpu = True
    if args.tpu and args.fp16:
        raise ValueError("Cannot combine --fp16 and --tpu, use --bf16 on TPUs")

    if getattr(args, "seed", None) is None:
        args.seed = 1  # default seed for training
        args.no_seed_provided = True
    else:
        args.no_seed_provided = False

    # Apply architecture configuration.
    if hasattr(args, "arch") and args.arch in ARCH_CONFIG_REGISTRY:
        ARCH_CONFIG_REGISTRY[args.arch](args)

    if parse_known:
        return args, extra
    else:
        return args


def get_parser(desc, default_task="translation"):
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args()
    utils.import_user_module(usr_args)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    gen_parser_from_dataclass(parser, CommonParams())

    from fairseq.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        parser.add_argument(
            "--" + registry_name.replace("_", "-"),
            default=REGISTRY["default"],
            choices=REGISTRY["registry"].keys(),
        )

    # Task definitions can be found under fairseq/tasks/
    from fairseq.tasks import TASK_REGISTRY

    parser.add_argument(
        "--task",
        metavar="TASK",
        default=default_task,
        choices=TASK_REGISTRY.keys(),
        help="task",
    )
    # fmt: on
    return parser


def add_preprocess_args(parser):
    group = parser.add_argument_group("Preprocessing")
    # fmt: off
    group.add_argument("-s", "--source-lang", default=None, metavar="SRC",
                       help="source language")
    group.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                       help="target language")
    group.add_argument("--trainpref", metavar="FP", default=None,
                       help="train file prefix")
    group.add_argument("--validpref", metavar="FP", default=None,
                       help="comma separated, valid file prefixes")
    group.add_argument("--testpref", metavar="FP", default=None,
                       help="comma separated, test file prefixes")
    group.add_argument("--align-suffix", metavar="FP", default=None,
                       help="alignment file suffix")
    group.add_argument("--destdir", metavar="DIR", default="data-bin",
                       help="destination dir")
    group.add_argument("--thresholdtgt", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--thresholdsrc", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--tgtdict", metavar="FP",
                       help="reuse given target dictionary")
    group.add_argument("--srcdict", metavar="FP",
                       help="reuse given source dictionary")
    group.add_argument("--nwordstgt", metavar="N", default=-1, type=int,
                       help="number of target words to retain")
    group.add_argument("--nwordssrc", metavar="N", default=-1, type=int,
                       help="number of source words to retain")
    group.add_argument("--alignfile", metavar="ALIGN", default=None,
                       help="an alignment file (optional)")
    parser.add_argument('--dataset-impl', metavar='FORMAT', default='mmap',
                        choices=get_available_dataset_impl(),
                        help='output dataset implementation')
    group.add_argument("--joined-dictionary", action="store_true",
                       help="Generate joined dictionary")
    group.add_argument("--only-source", action="store_true",
                       help="Only process the source language")
    group.add_argument("--padding-factor", metavar="N", default=8, type=int,
                       help="Pad dictionary size to be multiple of N")
    group.add_argument("--workers", metavar="N", default=1, type=int,
                       help="number of parallel workers")
    # fmt: on
    return parser


def add_dataset_args(parser, train=False, gen=False):
    group = parser.add_argument_group("dataset_data_loading")
    gen_parser_from_dataclass(group, DatasetParams())
    # fmt: on
    return group


def add_distributed_training_args(parser, default_world_size=None):
    group = parser.add_argument_group("distributed_training")
    if default_world_size is None:
        default_world_size = max(1, torch.cuda.device_count())
    gen_parser_from_dataclass(
        group, DistributedTrainingParams(distributed_world_size=default_world_size)
    )
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group("optimization")
    # fmt: off
    gen_parser_from_dataclass(group, OptimizationParams())
    # fmt: on
    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group("checkpoint")
    # fmt: off
    gen_parser_from_dataclass(group, CheckpointParams())
    # fmt: on
    return group


def add_common_eval_args(group):
    gen_parser_from_dataclass(group, CommonEvalParams())


def add_eval_lm_args(parser):
    group = parser.add_argument_group("LM Evaluation")
    add_common_eval_args(group)
    gen_parser_from_dataclass(group, EvalLMParams())


def add_generation_args(parser):
    group = parser.add_argument_group("Generation")
    add_common_eval_args(group)
    # fmt: off
    group.add_argument('--beam', default=5, type=int, metavar='N',
                       help='beam size')
    group.add_argument('--nbest', default=1, type=int, metavar='N',
                       help='number of hypotheses to output')
    group.add_argument('--max-len-a', default=0, type=float, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
    group.add_argument('--max-len-b', default=200, type=int, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
    group.add_argument('--min-len', default=1, type=float, metavar='N',
                       help=('minimum generation length'))
    group.add_argument('--match-source-len', default=False, action='store_true',
                       help=('generations should match the source length'))
    group.add_argument('--no-early-stop', action='store_true',
                       help='deprecated')
    group.add_argument('--unnormalized', action='store_true',
                       help='compare unnormalized hypothesis scores')
    group.add_argument('--no-beamable-mm', action='store_true',
                       help='don\'t use BeamableMM in attention layers')
    group.add_argument('--lenpen', default=1, type=float,
                       help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
    group.add_argument('--unkpen', default=0, type=float,
                       help='unknown word penalty: <0 produces more unks, >0 produces fewer')
    group.add_argument('--replace-unk', nargs='?', const=True, default=None,
                       help='perform unknown replacement (optionally with alignment dictionary)')
    group.add_argument('--sacrebleu', action='store_true',
                       help='score with sacrebleu')
    group.add_argument('--score-reference', action='store_true',
                       help='just score the reference translation')
    group.add_argument('--prefix-size', default=0, type=int, metavar='PS',
                       help='initialize generation by target prefix of given length')
    group.add_argument('--no-repeat-ngram-size', default=0, type=int, metavar='N',
                       help='ngram blocking such that this size ngram cannot be repeated in the generation')
    group.add_argument('--sampling', action='store_true',
                       help='sample hypotheses instead of using beam search')
    group.add_argument('--sampling-topk', default=-1, type=int, metavar='PS',
                       help='sample from top K likely next words instead of all words')
    group.add_argument('--sampling-topp', default=-1.0, type=float, metavar='PS',
                       help='sample from the smallest set whose cumulative probability mass exceeds p for next words')
    group.add_argument('--constraints', const="ordered", nargs="?", choices=["ordered", "unordered"],
                       help='enables lexically constrained decoding')
    group.add_argument('--temperature', default=1., type=float, metavar='N',
                       help='temperature for generation')
    group.add_argument('--diverse-beam-groups', default=-1, type=int, metavar='N',
                       help='number of groups for Diverse Beam Search')
    group.add_argument('--diverse-beam-strength', default=0.5, type=float, metavar='N',
                       help='strength of diversity penalty for Diverse Beam Search')
    group.add_argument('--diversity-rate', default=-1.0, type=float, metavar='N',
                       help='strength of diversity penalty for Diverse Siblings Search')
    group.add_argument('--print-alignment', action='store_true',
                       help='if set, uses attention feedback to compute and print alignment to source tokens')
    group.add_argument('--print-step', action='store_true')

    # arguments for iterative refinement generator
    group.add_argument('--iter-decode-eos-penalty', default=0.0, type=float, metavar='N',
                       help='if > 0.0, it penalized early-stopping in decoding.')
    group.add_argument('--iter-decode-max-iter', default=10, type=int, metavar='N',
                       help='maximum iterations for iterative refinement.')
    group.add_argument('--iter-decode-force-max-iter', action='store_true',
                       help='if set, run exact the maximum number of iterations without early stop')
    group.add_argument('--iter-decode-with-beam', default=1, type=int, metavar='N',
                       help='if > 1, model will generate translations varying by the lengths.')
    group.add_argument('--iter-decode-with-external-reranker', action='store_true',
                       help='if set, the last checkpoint are assumed to be a reranker to rescore the translations'),
    group.add_argument('--retain-iter-history', action='store_true',
                       help='if set, decoding returns the whole history of iterative refinement')
    group.add_argument('--retain-dropout', action='store_true',
                       help='Use dropout at inference time')
    group.add_argument('--retain-dropout-modules', default=None, nargs='+', type=str,
                       help='if set, only retain dropout for the specified modules; '
                            'if not set, then dropout will be retained for all modules')

    # special decoding format for advanced decoding.
    group.add_argument('--decoding-format', default=None, type=str, choices=['unigram', 'ensemble', 'vote', 'dp', 'bs'])
    # fmt: on
    return group


def add_interactive_args(parser):
    group = parser.add_argument_group("Interactive")
    # fmt: off
    group.add_argument('--buffer-size', default=0, type=int, metavar='N',
                       help='read this many sentences into a buffer before processing them')
    group.add_argument('--input', default='-', type=str, metavar='FILE',
                       help='file to read from; use - for stdin')
    # fmt: on


def add_model_args(parser):
    group = parser.add_argument_group("Model configuration")
    # fmt: off

    # Model definitions can be found under fairseq/models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    from fairseq.models import ARCH_MODEL_REGISTRY
    group.add_argument('--arch', '-a', metavar='ARCH',
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='model architecture')
    # fmt: on
    return group
