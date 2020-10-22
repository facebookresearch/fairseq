# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import options


def get_reranking_parser(default_task="translation"):
    parser = options.get_parser("Generation and reranking", default_task)
    add_reranking_args(parser)
    return parser


def get_tuning_parser(default_task="translation"):
    parser = options.get_parser("Reranking tuning", default_task)
    add_reranking_args(parser)
    add_tuning_args(parser)
    return parser


def add_reranking_args(parser):
    group = parser.add_argument_group("Reranking")
    # fmt: off
    group.add_argument('--score-model1', '-s1', type=str, metavar='FILE', required=True,
                       help='path to first model or ensemble of models for rescoring')
    group.add_argument('--score-model2', '-s2', type=str, metavar='FILE', required=False,
                       help='path to second model or ensemble of models for rescoring')
    group.add_argument('--num-rescore', '-n', type=int, metavar='N', default=10,
                       help='the number of candidate hypothesis to rescore')
    group.add_argument('-bz', '--batch-size', type=int, metavar='N', default=128,
                       help='batch size for generating the nbest list')
    group.add_argument('--gen-subset', default='test', metavar='SET', choices=['test', 'train', 'valid'],
                       help='data subset to generate (train, valid, test)')
    group.add_argument('--gen-model', default=None, metavar='FILE',
                       help='the model to generate translations')
    group.add_argument('-b1', '--backwards1', action='store_true',
                       help='whether or not the first model group is backwards')
    group.add_argument('-b2', '--backwards2', action='store_true',
                       help='whether or not the second model group is backwards')
    group.add_argument('-a', '--weight1', default=1, nargs='+', type=float,
                       help='the weight(s) of the first model')
    group.add_argument('-b', '--weight2', default=1, nargs='+', type=float,
                       help='the weight(s) of the second model, or the gen model if using nbest from interactive.py')
    group.add_argument('-c', '--weight3', default=1, nargs='+', type=float,
                       help='the weight(s) of the third model')

    # lm arguments
    group.add_argument('-lm', '--language-model', default=None, metavar='FILE',
                       help='language model for target language to rescore translations')
    group.add_argument('--lm-dict', default=None, metavar='FILE',
                       help='the dict of the language model for the target language')
    group.add_argument('--lm-name', default=None,
                       help='the name of the language model for the target language')
    group.add_argument('--lm-bpe-code', default=None, metavar='FILE',
                       help='the bpe code for the language model for the target language')
    group.add_argument('--data-dir-name', default=None,
                       help='name of data directory')
    group.add_argument('--lenpen', default=1, nargs='+', type=float,
                       help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
    group.add_argument('--score-dict-dir', default=None,
                       help='the directory with dictionaries for the scoring models')
    group.add_argument('--right-to-left1', action='store_true',
                       help='whether the first model group is a right to left model')
    group.add_argument('--right-to-left2', action='store_true',
                       help='whether the second model group is a right to left model')
    group.add_argument('--post-process', '--remove-bpe', default='@@ ',
                       help='the bpe symbol, used for the bitext and LM')
    group.add_argument('--prefix-len', default=None, type=int,
                       help='the length of the target prefix to use in rescoring (in terms of words wo bpe)')
    group.add_argument('--sampling', action='store_true',
                       help='use sampling instead of beam search for generating n best list')
    group.add_argument('--diff-bpe', action='store_true',
                       help='bpe for rescoring and nbest list not the same')
    group.add_argument('--rescore-bpe-code', default=None,
                       help='bpe code for rescoring models')
    group.add_argument('--nbest-list', default=None,
                       help='use predefined nbest list in interactive.py format')
    group.add_argument('--write-hypos', default=None,
                       help='filename prefix to write hypos to')
    group.add_argument('--ref-translation', default=None,
                       help='reference translation to use with nbest list from interactive.py')
    group.add_argument('--backwards-score-dict-dir', default=None,
                       help='the directory with dictionaries for the backwards model,'
                            'if None then it is assumed the fw and backwards models share dictionaries')

    # extra scaling args
    group.add_argument('--gen-model-name', default=None,
                       help='the name of the models that generated the nbest list')
    group.add_argument('--model1-name', default=None,
                       help='the name of the set for model1 group ')
    group.add_argument('--model2-name', default=None,
                       help='the name of the set for model2 group')
    group.add_argument('--shard-id', default=0, type=int,
                       help='the id of the shard to generate')
    group.add_argument('--num-shards', default=1, type=int,
                       help='the number of shards to generate across')
    group.add_argument('--all-shards', action='store_true',
                       help='use all shards')
    group.add_argument('--target-prefix-frac', default=None, type=float,
                       help='the fraction of the target prefix to use in rescoring (in terms of words wo bpe)')
    group.add_argument('--source-prefix-frac', default=None, type=float,
                       help='the fraction of the source prefix to use in rescoring (in terms of words wo bpe)')
    group.add_argument('--normalize', action='store_true',
                       help='whether to normalize by src and target len')
    # fmt: on
    return group


def add_tuning_args(parser):
    group = parser.add_argument_group("Tuning")

    group.add_argument(
        "--lower-bound",
        default=[-0.7],
        nargs="+",
        type=float,
        help="lower bound of search space",
    )
    group.add_argument(
        "--upper-bound",
        default=[3],
        nargs="+",
        type=float,
        help="upper bound of search space",
    )
    group.add_argument(
        "--tune-param",
        default=["lenpen"],
        nargs="+",
        choices=["lenpen", "weight1", "weight2", "weight3"],
        help="the parameter(s) to tune",
    )
    group.add_argument(
        "--tune-subset",
        default="valid",
        choices=["valid", "test", "train"],
        help="the subset to tune on ",
    )
    group.add_argument(
        "--num-trials",
        default=1000,
        type=int,
        help="number of trials to do for random search",
    )
    group.add_argument(
        "--share-weights", action="store_true", help="share weight2 and weight 3"
    )
    return group
