# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unittest
from fairseq import search
from fairseq.data import Dictionary
from fairseq.token_generation_constraints import *


def tensorize(constraints: List[List[int]]) -> torch.Tensor:
    return [torch.tensor(x) for x in constraints]


class TestNegativeConstraintState(unittest.TestCase):
    def setUp(self):
        # List of Tuples of contraint set, input sequence, negative tokens per step
        # The position of index 0 of negative tokens sets is for root
        self.sequences = [
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [1, 2, 3, 4, 5, 6, 7],
                [{1}, {1, 3, 4}, {1, 3}, {1}, {1, 5}, {1}, {1, 7}, {1}],
            ),
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [7, 6, 5, 4, 3, 2, 1],
                [{1}, {1}, {1}, {1}, {1, 5}, {1}, {1}, {1, 3, 4}],
            ),
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [4, 5, 99, 6],
                [{1}, {1, 5}, {1}, {1}, {1}],
            ),
            (
                tensorize([[1], [2], [3]]),
                [1, 2, 99, 3],
                [{1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}],
            ),
        ]

    def test_sequences(self):
        for constraints, tokens, expected_negative_tokens in self.sequences:
            state = UnorderedConstraintState.create(pack_constraints([constraints])[0])
            root_negative_tokens = state.negative_tokens()
            assert root_negative_tokens == expected_negative_tokens[0],\
                f"TEST( Root ) GOT: {root_negative_tokens} WANTED: {expected_negative_tokens[0]}"
            for token, step_expected in zip(tokens, expected_negative_tokens[1:] ):
                state = state.advance(token)
                node_negative_tokens = state.negative_tokens()
                assert node_negative_tokens == step_expected,\
                    f"TEST( {token} ) GOT: {node_negative_tokens} WANTED: {step_expected}"


class TestNegativeConstraintBeam(unittest.TestCase):

    def setUp(self):
        # bsz = 1
        self.beam_size = 2
        self.cand_size = 2 * self.beam_size
        # The process is :
        # step1. use test_constraints to init_constraints
        # step2. select corresponding candidates according to cand_beam_idx[:, step]
        # step3. according to step2's selection, update corresponding position of negative constraints state
        # using tokens of cand_matrix[:, step]
        # step4. select every step active candidates according to active_beam_idx[:, step]
        # This setup will generate the beam history as below:
        # [
        #    [101, 102, 103, 104, 999]
        #    [103, 999, 102, 101, 102]
        #    [101, 102, 103, 104, 101]
        #    [101, 102, 103, 104, 103]
        # ]
        self.positive_constraints = tensorize([[-1]])
        self.negative_constraints = tensorize([[101, 102, 103], [101, 103], [101, 104], [102, 104]])

        self.cand_matrix = torch.tensor([[101, 102, 103, 104, 999],
                                         [103, 999, 102, 101, 102],
                                         [101, 102, 103, 102, 101],
                                         [103, 999, 103, 103, 103]])

        self.steps = self.cand_matrix.size(1)

        self.cand_beam_idx = torch.tensor([[0, 2, 0, 0, 0],
                                           [1, 1, 1, 1, 1],
                                           [0, 0, 2, 0, 2],
                                           [0, 2, 2, 0, 3]])

        self.active_beam_idx = torch.tensor([[0, 0, 0, 0, 2],
                                             [1, 1, 1, 1, 3]])

        self.expected = [{(0, 103), (0, 104)},
                         {(0, 103)},
                         {(1, 104)},
                         {(1, 104), (1, 103)},
                         {(0, 103), (0, 104)}]

    def test_beam(self):
        pseudo_dict = Dictionary()
        toy_search = search.LexicallyConstrainedBeamSearch(pseudo_dict, "unordered")
        toy_search.init_constraints(
            pack_constraints([self.positive_constraints]),
            pack_constraints([self.negative_constraints]),
            self.beam_size,
            self.cand_size
        )

        for step in range(self.steps):
            toy_search.update_negative_constraints(torch.unsqueeze(self.cand_matrix[:, step], 0),
                                               torch.unsqueeze(self.cand_beam_idx[:, step], 0))
            result = set([tuple(i) for i in toy_search.get_negative_tokens(torch.unsqueeze(self.active_beam_idx[:, step], 0))])
            assert (
                self.expected[step] == result
            ), f"TEST at step ({step}) GOT: {result} WANTED: {self.expected[step]}"


if __name__ == "__main__":
    unittest.main()
