# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import numpy as np


class WordNoising(object):
    """Generate a noisy version of a sentence, without changing words themselves."""
    def __init__(self, dictionary, bpe_cont_marker="@@"):
        self.dictionary = dictionary
        self.bpe_end = np.array([
            not self.dictionary[i].endswith(bpe_cont_marker)
            for i in range(len(self.dictionary))
        ])

    def noising(self, x, lengths, noising_prob=0.0):
        raise NotImplementedError()

    def _get_bpe_word_idx(self, x):
        # x: (T x B)
        bpe_end = self.bpe_end[x]
        # do a reduce front sum to generate word ids
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx
        return word_idx


class WordDropout(WordNoising):
    """Randomly drop input words. If not passing blank_idx (default is None),
    then dropped words will be removed. Otherwise, it will be replaced by the
    blank_idx."""

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def noising(self, x, lengths, dropout_prob=0.1, blank_idx=None):
        # x: (T x B), lengths: B
        if dropout_prob == 0:
            return x, lengths

        assert 0 < dropout_prob < 1

        # be sure to drop entire words
        word_idx = self._get_bpe_word_idx(x)
        sentences = []
        modified_lengths = []
        for i in range(lengths.size(0)):
            # Since dropout probabilities need to apply over non-pad tokens,
            # it is not trivial to generate the keep mask without consider
            # input lengths; otherwise, this could be done outside the loop
            keep = np.random.rand(lengths[i] - 1) >= dropout_prob
            # ith example: [x0, x1, ..., eos, pad, ..., pad]
            assert x[lengths[i] - 1, i] == self.dictionary.eos()
            words = x[:lengths[i], i].tolist()

            # TODO: speed up the following loop
            # drop words from the input according to keep
            new_s = [
                w if keep[word_idx[j, i]] else blank_idx
                for j, w in enumerate(words)
            ]
            new_s = [w for w in new_s if w is not None]
            # we need to have at least one word in the sentence (more than the
            # start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(0, len(words))])
            assert (
                len(new_s) >= 2
                and new_s[-1] == self.dictionary.eos()
            ), "New sentence is invalid."
            sentences.append(new_s)
            modified_lengths.append(len(new_s))
        # re-construct input
        modified_lengths = torch.LongTensor(modified_lengths)
        modified_x = torch.LongTensor(
            modified_lengths.max(),
            modified_lengths.size(0)
        ).fill_(self.dictionary.pad())
        for i in range(modified_lengths.size(0)):
            modified_x[:modified_lengths[i], i].copy_(torch.LongTensor(sentences[i]))

        return modified_x, modified_lengths


class WordShuffle(WordNoising):
    """Shuffle words by no more than k positions."""

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def noising(self, x, lengths, max_shuffle_distance=3):
        # x: (T x B), lengths: B
        if max_shuffle_distance == 0:
            return x, lengths

        # max_shuffle_distance < 1 will return the same sequence
        assert max_shuffle_distance > 1

        # define noise word scores
        noise = np.random.uniform(
            0,
            max_shuffle_distance,
            size=(x.size(0) - 1, x.size(1)),
        )
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        word_idx = self._get_bpe_word_idx(x)

        x2 = x.clone()
        for i in range(lengths.size(0)):
            # generate a random permutation
            scores = word_idx[:lengths[i] - 1, i] + noise[word_idx[:lengths[i] - 1, i], i]
            # ensure no reordering inside a word
            scores += 1e-6 * np.arange(lengths[i] - 1)
            permutation = scores.argsort()
            # shuffle words
            x2[:lengths[i] - 1, i].copy_(
                x2[:lengths[i] - 1, i][torch.from_numpy(permutation)]
            )
        return x2, lengths
