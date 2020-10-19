# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random

import numpy as np
from fairseq import options

from . import rerank, rerank_options


def random_search(args):
    param_values = []
    tuneable_parameters = ["lenpen", "weight1", "weight2", "weight3"]
    initial_params = [args.lenpen, args.weight1, args.weight2, args.weight3]
    for i, elem in enumerate(initial_params):
        if type(elem) is not list:
            initial_params[i] = [elem]
        else:
            initial_params[i] = elem

    tune_parameters = args.tune_param.copy()
    for i in range(len(args.tune_param)):
        assert args.upper_bound[i] >= args.lower_bound[i]
        index = tuneable_parameters.index(args.tune_param[i])
        del tuneable_parameters[index]
        del initial_params[index]

    tune_parameters += tuneable_parameters
    param_values += initial_params
    random.seed(args.seed)

    random_params = np.array(
        [
            [
                random.uniform(args.lower_bound[i], args.upper_bound[i])
                for i in range(len(args.tune_param))
            ]
            for k in range(args.num_trials)
        ]
    )
    set_params = np.array(
        [
            [initial_params[i][0] for i in range(len(tuneable_parameters))]
            for k in range(args.num_trials)
        ]
    )
    random_params = np.concatenate((random_params, set_params), 1)

    rerank_args = vars(args).copy()
    if args.nbest_list:
        rerank_args["gen_subset"] = "test"
    else:
        rerank_args["gen_subset"] = args.tune_subset

    for k in range(len(tune_parameters)):
        rerank_args[tune_parameters[k]] = list(random_params[:, k])

    if args.share_weights:
        k = tune_parameters.index("weight2")
        rerank_args["weight3"] = list(random_params[:, k])

    rerank_args = argparse.Namespace(**rerank_args)
    best_lenpen, best_weight1, best_weight2, best_weight3, best_score = rerank.rerank(
        rerank_args
    )
    rerank_args = vars(args).copy()
    rerank_args["lenpen"] = [best_lenpen]
    rerank_args["weight1"] = [best_weight1]
    rerank_args["weight2"] = [best_weight2]
    rerank_args["weight3"] = [best_weight3]

    # write the hypothesis from the valid set from the best trial

    if args.gen_subset != "valid":
        rerank_args["gen_subset"] = "valid"
        rerank_args = argparse.Namespace(**rerank_args)
        rerank.rerank(rerank_args)

    # test with the best hyperparameters on gen subset
    rerank_args = vars(args).copy()
    rerank_args["gen_subset"] = args.gen_subset
    rerank_args["lenpen"] = [best_lenpen]
    rerank_args["weight1"] = [best_weight1]
    rerank_args["weight2"] = [best_weight2]
    rerank_args["weight3"] = [best_weight3]
    rerank_args = argparse.Namespace(**rerank_args)
    rerank.rerank(rerank_args)


def cli_main():
    parser = rerank_options.get_tuning_parser()
    args = options.parse_args_and_arch(parser)

    random_search(args)


if __name__ == "__main__":
    cli_main()
