# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest

import torch
from fairseq.token_generation_constraints import *


def tensorize(constraints: List[List[int]]) -> torch.Tensor:
    return [torch.tensor(x) for x in constraints]


class TestHelperRoutines(unittest.TestCase):
    def setUp(self):
        self.examples = [
            ([[]], torch.tensor([[0]])),
            ([[], []], torch.tensor([[0], [0]])),
            ([[torch.tensor([1, 2])], []], torch.tensor([[1, 1, 2, 0], [0, 0, 0, 0]])),
            (
                [
                    [
                        torch.tensor([3, 1, 2]),
                        torch.tensor([3]),
                        torch.tensor([4, 5, 6, 7]),
                    ],
                    [],
                    [torch.tensor([1, 8, 9, 10, 1, 4, 11, 12])],
                ],
                torch.tensor(
                    [
                        [3, 3, 1, 2, 0, 3, 0, 4, 5, 6, 7, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 8, 9, 10, 1, 4, 11, 12, 0, 0, 0],
                    ]
                ),
            ),
        ]

    def test_packing(self):
        """Ensures the list of lists of tensors gets packed correctly."""
        for batch_constraints, expected_tensor in self.examples:
            packed = pack_constraints(batch_constraints)
            assert torch.equal(packed, expected_tensor)


class TestUnorderedConstraintState(unittest.TestCase):
    def setUp(self):
        # Tuples of (contraint set, expected printed graph, token counts per node)
        self.examples = [
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                "([None].False#6 ([1].True#4 ([2].False#1 [3].True#1) [3].True#1 [4].True#1) ([4].False#2 ([5].True#2 ([6].False#1 [7].True#1))))",
                {1: 4, 2: 1, 3: 2, 4: 3, 5: 2, 6: 1, 7: 1},
            ),
            ([], "[None].False#0", {}),
            (tensorize([[0]]), "([None].False#1 [0].True#1)", {0: 1}),
            (
                tensorize([[100000, 1, 2, 3, 4, 5]]),
                "([None].False#1 ([100000].False#1 ([1].False#1 ([2].False#1 ([3].False#1 ([4].False#1 [5].True#1))))))",
                {100000: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
            ),
            (
                tensorize([[1, 2], [1, 2]]),
                "([None].False#2 ([1].False#2 [2].True#2))",
                {1: 2, 2: 2},
            ),
            (
                tensorize([[1, 2], [3, 4]]),
                "([None].False#2 ([1].False#1 [2].True#1) ([3].False#1 [4].True#1))",
                {1: 1, 2: 1, 3: 1, 4: 1},
            ),
        ]

        self.sequences = [
            (
                self.examples[0][0],
                [],
                {"bank": 0, "num_completed": 0, "finished": False, "is_root": True},
            ),
            (
                self.examples[0][0],
                [1, 2],
                {"bank": 2, "num_completed": 0, "finished": False, "is_root": False},
            ),
            (
                self.examples[0][0],
                [1, 2, 94],
                {"bank": 1, "num_completed": 1, "finished": False, "is_root": True},
            ),
            (
                self.examples[0][0],
                [1, 3, 999, 1, 4],
                {"bank": 4, "num_completed": 2, "finished": False, "is_root": False},
            ),
            (
                self.examples[0][0],
                [1, 3, 999, 1, 4, 999],
                {"bank": 4, "num_completed": 2, "finished": False, "is_root": True},
            ),
            (
                self.examples[0][0],
                [4, 5, 6, 8],
                {"bank": 2, "num_completed": 1, "finished": False, "is_root": True},
            ),
            (
                self.examples[0][0],
                # Tricky, because in last three, goes down [1->4] branch, could miss [1] and [4->5]
                # [[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]],
                [1, 2, 3, 1, 3, 1, 4, 4, 5, 6, 7, 1, 4, 5],
                {"bank": 14, "num_completed": 6, "finished": True, "is_root": False},
            ),
            (
                self.examples[0][0],
                [1, 2, 3, 999, 1, 3, 1, 4, 4, 5, 6, 7, 1, 4, 5, 117],
                {"bank": 14, "num_completed": 6, "finished": True, "is_root": True},
            ),
            (
                tensorize([[1], [2, 3]]),
                # Should not be able to get credit for entering 1 a second time
                [1, 1],
                {"bank": 1, "num_completed": 1, "finished": False, "is_root": True},
            ),
            (
                self.examples[4][0],
                [1, 2, 1, 2],
                {"bank": 4, "num_completed": 2, "finished": True, "is_root": False},
            ),
            (
                self.examples[4][0],
                [1, 2, 1, 2, 1],
                {"bank": 4, "num_completed": 2, "finished": True, "is_root": True},
            ),
            (
                self.examples[5][0],
                [1, 2, 3, 4, 5],
                {"bank": 4, "num_completed": 2, "finished": True, "is_root": True},
            ),
        ]

    def test_graphs(self):
        """
        Test whether unordered graph systems are created correctly.
        """
        for example in self.examples:
            constraints, expected, gold_counts = example
            c = ConstraintNode.create(constraints)
            assert (
                ConstraintNode.print_graph(c) == expected
            ), f"got {ConstraintNode.print_graph(c)}, expected {expected}"
            assert (
                c.token_counts() == gold_counts
            ), f"{c} got {c.token_counts()} wanted {gold_counts}"

    def test_next_tokens(self):
        """
        Tests that the set of next tokens is correct.
        """
        for example in self.examples:
            constraints, expected, gold_counts = example
            root = ConstraintNode.create(constraints)

            root_tokens = set(root.children.keys())
            for sequence in constraints:
                state = UnorderedConstraintState(root)
                for token in sequence:
                    all_tokens = root_tokens.union(state.node.children.keys())
                    assert (
                        all_tokens == state.next_tokens()
                    ), f"ALL {all_tokens} NEXT {state.next_tokens()}"
                    state = state.advance(token)

    def test_sequences(self):
        for constraints, tokens, expected in self.sequences:
            state = UnorderedConstraintState.create(pack_constraints([constraints])[0])
            for token in tokens:
                state = state.advance(token)
            result = {}
            for attr in expected.keys():
                result[attr] = getattr(state, attr)

            assert (
                result == expected
            ), f"TEST({tokens}) GOT: {result} WANTED: {expected}"


class TestOrderedConstraintState(unittest.TestCase):
    def setUp(self):
        self.sequences = [
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [],
                {"bank": 0, "num_completed": 0, "finished": False, "is_root": True},
            ),
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [1, 2],
                {"bank": 2, "num_completed": 0, "finished": False, "is_root": False},
            ),
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [1, 2, 94],
                {"bank": 0, "num_completed": 0, "finished": False, "is_root": True},
            ),
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [1, 3, 999, 1, 4],
                {"bank": 0, "num_completed": 0, "finished": False, "is_root": True},
            ),
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [1, 2, 3, 999, 999],
                {"bank": 3, "num_completed": 1, "finished": False, "is_root": False},
            ),
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [1, 2, 3, 77, 1, 3, 1],
                {"bank": 6, "num_completed": 2, "finished": False, "is_root": False},
            ),
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [1, 2, 3, 1, 3, 1, 4, 4, 5, 6, 7, 1, 4, 5],
                {"bank": 14, "num_completed": 6, "finished": True, "is_root": False},
            ),
            (
                tensorize([[1, 2, 3], [1, 3], [1, 4], [4, 5, 6, 7], [1], [4, 5]]),
                [1, 2, 999, 1, 2, 3, 999, 1, 3, 1, 4, 4, 5, 6, 7, 1, 4, 5, 117],
                {"bank": 14, "num_completed": 6, "finished": True, "is_root": False},
            ),
            (
                tensorize([[1], [2, 3]]),
                [1, 1],
                {"bank": 1, "num_completed": 1, "finished": False, "is_root": False},
            ),
            (
                tensorize([[1, 2], [1, 2]]),
                [1, 2, 1, 2],
                {"bank": 4, "num_completed": 2, "finished": True, "is_root": False},
            ),
            (
                tensorize([[1, 2], [1, 2]]),
                [1, 2, 1, 2, 1],
                {"bank": 4, "num_completed": 2, "finished": True, "is_root": False},
            ),
            (
                tensorize([[1, 2], [3, 4]]),
                [1, 2, 3, 4, 5],
                {"bank": 4, "num_completed": 2, "finished": True, "is_root": False},
            ),
        ]

    def test_sequences(self):
        for i, (constraints, tokens, expected) in enumerate(self.sequences):
            state = OrderedConstraintState.create(pack_constraints([constraints])[0])
            for token in tokens:
                state = state.advance(token)
            result = {}
            for attr in expected.keys():
                result[attr] = getattr(state, attr)
            assert (
                result == expected
            ), f"TEST({tokens}) GOT: {result} WANTED: {expected}"


if __name__ == "__main__":
    unittest.main()
