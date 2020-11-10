# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Implements tracking of constraints for a beam item.

A list of constraints is given as a list of one or more token
sequences, each of length at least one token. For example, for an input sentence

> Die maschinelle Übersetzung ist schwer zu kontrollieren.

We could have the constraints:
* to influence
* hard

There are two implementations:
* OrderedConstraintState: Tracks progress through an ordered list of multitoken constraints.
* UnorderedConstraintState: Tracks progress through an unordered list of multitoken constraints.

The difference is that in the first, the constraints are assumed to be
in order; the algorithm will permit zero or more tokens between them.
In the second, the constraints are not ordered, so many orderings will
be explored.

The same sequence can be present any number of times, and will appear
that many times in the output.
"""

from collections import Counter
from typing import List, Optional, Set, Tuple

import torch


class ConstraintState:
    def __init__(self):
        pass


def pack_constraints(batch_constraints: List[List[torch.Tensor]]) -> torch.Tensor:
    """Takes a list of list of constraints in tensor form (a list of
    tensor constraints for each sentence) and transforms it into a
    packed Tensor. For example, here is a batch of size 3 with 3, 0,
    and 1 constraints:

        [ [ [3 1 2], [3], [4 5 6 7], ]
          [],
          [ [1 8 9 10 1 4 11 12], ]
        ]

    Its corresponding packed structure is:

        [ [ 3  3  1  2  0  3  0  4  5  6  7  0],
          [ 0  0  0  0  0  0  0  0  0  0  0  0],
          [ 1  1  8  9 10  1  4 11 12  0  0  0] ]

    The packed tensor has shape (batch size, maxlen), where
    maxlen is defined below. Each row contains concatenated
    constraint tokens for that sentence, with 0 appended after
    each constraint. The first item in each row is the number
    of constraints for that sentence. So maxlen is the maximum
    of

    (number of constraints) + (sum length of constraints) + 1.

    across all sentences in the batch.
    """
    # The maximum word length of concatenated constraints for any sentence
    max_constraints_len = 1
    for sentence_constraints in batch_constraints:
        if len(sentence_constraints):
            # number of constraints, plus sum of constrain lens, plus a zero after each
            constraints_len = (
                1
                + sum([c.size(0) for c in sentence_constraints])
                + len(sentence_constraints)
            )
            max_constraints_len = max(max_constraints_len, constraints_len)

    batch_size = len(batch_constraints)
    constraints_tensor = torch.zeros((batch_size, max_constraints_len)).long()
    for i, sentence_constraints in enumerate(batch_constraints):
        constraints_tensor[i, 0] = len(sentence_constraints)
        offset = 1
        for j, constraint in enumerate(sentence_constraints):
            this_len = constraint.size(0)
            constraints_tensor[i, offset : offset + this_len] = constraint
            offset += this_len + 1

    return constraints_tensor.long()


def unpack_constraints(constraint_tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Transforms *one row* of a packed constraint tensor (e.g., for one
    sentence in the batch) into a list of constraint tensors.
    """
    constraint_list = []
    num_constraints = constraint_tensor[0]
    constraints = constraint_tensor.tolist()
    offset = 1
    for i in range(num_constraints):
        where = constraints.index(0, offset)
        constraint_list.append(constraint_tensor[offset:where])
        offset = where + 1

    return constraint_list


class ConstraintNode:
    """
    Represents a node in a trie managing unordered constraints.
    """

    def __init__(self, token: int = None, parent=None):
        # The token associate with this node (None for the root)
        self.token = int(token) if token is not None else None
        # The parent (None at the root)
        self.parent = parent
        # Whether this node is a completed constraint
        self.terminal = 0
        # List of child nodes
        self.children = {}

        # The cumulative number of constraints from this point in the
        # trie forward
        self.num_constraints = 0

    @property
    def id(self):
        return self.token

    def __str__(self):
        term = self.terminal != 0
        return f"[{self.token}].{term}#{self.num_constraints}"

    def __getitem__(self, key: int):
        return self.children.get(key, None)

    def next_tokens(self) -> Set[int]:
        """The set of child labels."""
        return set(self.children.keys())

    @staticmethod
    def create(constraints: List[List[int]]):
        root = ConstraintNode()
        for sequence in constraints:
            root.add_sequence(sequence)

        return root

    @staticmethod
    def print_graph(node: "ConstraintNode"):
        if len(node.children) == 0:
            return str(node)
        else:
            s = f"({node}"
            for child in node.children.values():
                s += " " + ConstraintNode.print_graph(child)
            s += ")"
            return s

    def token_counts(self) -> Counter:
        """Returns a counter of the number of times each token is used
        in a constraint.
        """
        token_counts = Counter()
        kids = list(self.children.values())
        while len(kids) > 0:
            kid = kids.pop()
            token_counts[kid.id] += kid.num_constraints
            kids += list(kid.children.values())

        return token_counts

    def tokens(self) -> Set[int]:
        """Returns the set of tokens in constraints."""
        return set(self.token_counts().keys())

    def add_sequence(self, sequence: List[int]):
        """Adds a constraint, represented as a list of integers, to
        the trie."""
        assert len(sequence) > 0

        token = int(sequence[0])
        if token not in self.children:
            self.children[token] = ConstraintNode(token, parent=self)

        node = self.children[token]
        if len(sequence) == 1:
            node.terminal += 1
            node.num_constraints += 1
            parent = node.parent
            while parent is not None:
                parent.num_constraints += 1
                parent = parent.parent
        else:
            node.add_sequence(sequence[1:])


class UnorderedConstraintState(ConstraintState):
    """
    Records progress through the set of constraints for each item in the beam
    using a trie.
    """

    def __init__(self, node: ConstraintNode, copy_from: "ConstraintState" = None):
        self.node = node

        if copy_from is None:
            # The root node
            self.root = node
            # The set of states in the graph that have been completed
            self.completed = Counter()
            # The...
            self.generated = Counter()
            # The list of tokens we need to generate
            self.needed_tokens = self.root.tokens()
        else:
            self.completed = Counter(copy_from.completed)
            self.generated = Counter(copy_from.generated)
            self.root = copy_from.root

        # Mark the node as generated
        if self.node != self.root:
            self.generated[node] += 1

    @staticmethod
    def create(constraint_tensor: torch.Tensor):
        constraint_list = unpack_constraints(constraint_tensor)
        constraint_trie_root = ConstraintNode.create(constraint_list)
        return UnorderedConstraintState(constraint_trie_root)

    def __str__(self):
        gen_str = ",".join([str(node) for node in self.generated])
        return f"{self.name}/{self.bank}({gen_str})x{self.num_completed}"

    def __copy__(self):
        copied_state = UnorderedConstraintState(self.node, copy_from=self)
        return copied_state

    def copy(self):
        return self.__copy__()

    @property
    def name(self):
        if self.node.id is None:
            return "ROOT"
        else:
            return str(self.node.id)

    @property
    def is_root(self):
        return self.node == self.root

    @property
    def bank(self):
        return sum(self.generated.values())

    @property
    def num_completed(self):
        """The number of constraints (not constraint tokens) that are completed.
        In addition to the already-completed states, we need to account for the
        current state, which might get marked as completed when another token
        is generated.
        """
        in_final = self.node.terminal and self.completed[self.node] < self.node.terminal
        return sum(self.completed.values()) + in_final

    @property
    def finished(self):
        return self.root.num_constraints - self.num_completed == 0

    @property
    def token_counts(self):
        return self.root.token_counts()

    @property
    def tokens(self):
        return self.root.tokens()

    @property
    def num_constraint_tokens(self):
        return sum(self.token_counts.values())

    def next_tokens(self) -> Set[int]:
        """Returns the list of tokens that could come next.
        These are (a) all tokens extending the root state and, for
        non-root states, additionally all tokens extending the current
        state."""

        if self.node != self.root:
            return self.root.next_tokens().union(self.node.next_tokens())
        else:
            return self.root.next_tokens()

    def advance(self, token: int):
        """Reads in a token and advances the state. Here's how it works.

        We can advance to the next state if:
        - there is a matching child
        - its path isn't blocked

        A path is blocked when all constraints that are descendants of
        that node have already been generated, in the current state.

        If we are not able to advance from the current state, we "fall
        off the graph" and return to the root state. There, we again
        try to advance, checking the same criteria.

        In any case, when falling off the graph, we need to do some
        bookkeeping. We:
        - check whether any constraints were met (all prefixes of
          current state)
        - if one is found, mark it as completed
        - adjust visited nodes accordingly
        """
        token = int(token)

        next_state = None
        child = self.node[token]
        if child is not None and self.generated[child] < child.num_constraints:
            next_state = UnorderedConstraintState(child, copy_from=self)

        def rewind():
            """If we're mid-trie and an "illegal" token is chosen next, we need
            to reset our state to the root state. However, along the way, we need
            to check whether a prefix of the current trie state represents a state
            we could mark as completed.
            """
            node = self.node
            while node != self.root:
                if node.terminal and self.completed[node] < node.terminal:
                    next_state.completed[node] += 1
                    return

                next_state.generated[node] -= 1
                node = node.parent

        # Fall off the graph, check the root
        if next_state is None and token in self.root.next_tokens():
            child = self.root[token]
            # We can only traverse this edge if it's not saturated
            if self.generated[child] < child.num_constraints:
                next_state = UnorderedConstraintState(child, copy_from=self)
            else:
                next_state = UnorderedConstraintState(self.root, copy_from=self)

            # Rewind
            rewind()

        elif next_state is None:
            next_state = UnorderedConstraintState(self.root, copy_from=self)
            # Rewind
            rewind()

        return next_state


class ConstraintSequence:
    def __init__(self, sequences: List[List[int]]):
        """Represents a set of possibly multitoken constraints by
        concatenating them and internally recording the end points.
        """
        self.sequences = []
        self.endpoints = []
        self.num_tokens = 0
        self.tokens = set()
        for sequence in sequences:
            for token in sequence:
                self.tokens.add(token)
            self.num_tokens += len(sequence)
            self.endpoints += [False for x in range(len(sequence) - 1)] + [True]
            self.sequences += sequence

    def __getitem__(self, key: int):
        return self.sequences[key]

    def __len__(self):
        return len(self.sequences)

    def __str__(self):
        return str(self.sequences)


class OrderedConstraintState(ConstraintState):
    """
    Records progress through the set of linear nonbranching constraints with gaps.
    """

    def __init__(self, sequence: ConstraintSequence, state: int = -1):
        self.sequence = sequence
        self.state = state

    @staticmethod
    def create(constraint_tensor: torch.Tensor):
        constraint_list = unpack_constraints(constraint_tensor)
        return OrderedConstraintState(ConstraintSequence(constraint_list), -1)

    def __str__(self):
        return f"{self.state}/{self.bank}x{self.num_completed}"

    def __copy__(self):
        return OrderedConstraintState(self.sequence, self.state)

    def copy(self):
        return self.__copy__()

    @property
    def num_completed(self):
        if self.state == -1:
            return 0
        count = len(
            list(filter(lambda x: x, self.sequence.endpoints[0 : self.state + 1]))
        )
        return count

    @property
    def is_root(self):
        return self.state == -1

    @property
    def name(self):
        if self.state == -1:
            return "ROOT"
        else:
            return str(self.sequence[self.state])

    @property
    def bank(self) -> int:
        return self.state + 1

    @property
    def finished(self):
        return self.state + 1 == len(self.sequence)

    @property
    def token_counts(self):
        return self.sequence.token_counts()

    @property
    def tokens(self):
        return self.sequence.tokens

    @property
    def num_constraint_tokens(self):
        return sum(self.token_counts.values())

    def next_tokens(self) -> Set[int]:
        """Returns the list of tokens that could come next.
        These are (a) all tokens extending the root state and, for
        non-root states, additionally all tokens extending the current
        state."""

        tokens = set()
        if self.state > 0:
            tokens.add(self.sequence[0])
        if not self.finished:
            tokens.add(self.sequence[self.state + 1])
        return tokens

    def advance(self, token: int):
        """Reads in a token and advances the state. Here's how it works.

        We can advance to the next state if:
        - there is a matching child
        - its path isn't blocked

        A path is blocked when all constraints that are descendants of
        that node have already been generated, in the current state.

        If we are not able to advance from the current state, we "fall
        off the graph" and return to the root state. There, we again
        try to advance, checking the same criteria.

        In any case, when falling off the graph, we need to do some
        bookkeeping. We:
        - check whether any constraints were met (all prefixes of
          current state)
        - if one is found, mark it as completed
        - adjust visited nodes accordingly
        """
        token = int(token)
        # print(f"{self} ADVANCE({token}) {self.sequence} -> ", end="")

        if self.finished:
            # Accept anything
            next_state = self.copy()

        elif self.sequence[self.state + 1] == token:
            # Advance to the next token
            next_state = OrderedConstraintState(self.sequence, self.state + 1)

        elif self.sequence.endpoints[self.state]:
            # Accept anything between constraints (*)
            next_state = self.copy()

        elif token == self.sequence[0]:
            # Start over having generated the first token
            next_state = OrderedConstraintState(self.sequence, 0)
        else:
            # Start over from the root
            next_state = OrderedConstraintState(self.sequence, -1)

        return next_state
