#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Replabel transforms for use with flashlight's ASG criterion.
"""


def replabel_symbol(i):
    """
    Replabel symbols used in flashlight, currently just "1", "2", ...
    This prevents training with numeral tokens, so this might change in the future
    """
    return str(i)


def pack_replabels(tokens, dictionary, max_reps):
    """
    Pack a token sequence so that repeated symbols are replaced by replabels
    """
    if len(tokens) == 0 or max_reps <= 0:
        return tokens

    replabel_value_to_idx = [0] * (max_reps + 1)
    for i in range(1, max_reps + 1):
        replabel_value_to_idx[i] = dictionary.index(replabel_symbol(i))

    result = []
    prev_token = -1
    num_reps = 0
    for token in tokens:
        if token == prev_token and num_reps < max_reps:
            num_reps += 1
        else:
            if num_reps > 0:
                result.append(replabel_value_to_idx[num_reps])
                num_reps = 0
            result.append(token)
            prev_token = token
    if num_reps > 0:
        result.append(replabel_value_to_idx[num_reps])
    return result


def unpack_replabels(tokens, dictionary, max_reps):
    """
    Unpack a token sequence so that replabels are replaced by repeated symbols
    """
    if len(tokens) == 0 or max_reps <= 0:
        return tokens

    replabel_idx_to_value = {}
    for i in range(1, max_reps + 1):
        replabel_idx_to_value[dictionary.index(replabel_symbol(i))] = i

    result = []
    prev_token = -1
    for token in tokens:
        try:
            for _ in range(replabel_idx_to_value[token]):
                result.append(prev_token)
            prev_token = -1
        except KeyError:
            result.append(token)
            prev_token = token
    return result
