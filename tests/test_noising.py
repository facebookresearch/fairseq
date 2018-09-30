# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import unittest

from fairseq.data import data_utils, Dictionary, noising


class TestDataNoising(unittest.TestCase):
    def _get_test_data(self):
        vocab = Dictionary()
        vocab.add_symbol("he@@")
        vocab.add_symbol("llo")
        vocab.add_symbol("how")
        vocab.add_symbol("are")
        vocab.add_symbol("y@@")
        vocab.add_symbol("ou")
        vocab.add_symbol("n@@")
        vocab.add_symbol("ew")
        vocab.add_symbol("or@@")
        vocab.add_symbol("k")

        src_tokens = [
            ["he@@", "llo", "n@@", "ew", "y@@", "or@@", "k"],
            ["how", "are", "y@@", "ou"],
        ]
        src_len = [len(x) for x in src_tokens]
        x = torch.LongTensor(len(src_tokens), max(src_len) + 1).fill_(vocab.pad())
        for i in range(len(src_tokens)):
            for j in range(len(src_tokens[i])):
                x[i][j] = vocab.index(src_tokens[i][j])
            x[i][j + 1] = vocab.eos()

        x = x.transpose(1, 0)
        return vocab, x, torch.LongTensor([i + 1 for i in src_len])

    def test_word_dropout(self):
        vocab, x, x_len = self._get_test_data()

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2)
            # Expect only the first word (2 bpe tokens) of the first example
            # was dropped out
            self.assertEqual(x_len[0] - 2, l_noised[0])
            for i in range(l_noised[0]):
                self.assertEqual(x_noised[i][0], x[i+2][0])

    def test_word_blank(self):
        vocab, x, x_len = self._get_test_data()

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2, vocab.unk())
            # Expect only the first word (2 bpe tokens) of the first example
            # was blanked out
            self.assertEqual(x_len[0], l_noised[0])
            for i in range(l_noised[0]):
                if i < 2:
                    self.assertEqual(x_noised[i][0], vocab.unk())
                else:
                    self.assertEqual(x_noised[i][0], x[i][0])

    def test_word_shuffle(self):
        vocab, x, x_len = self._get_test_data()

        with data_utils.numpy_seed(1234):
            word_shuffle = noising.WordShuffle(vocab)

            x_noised, l_noised = word_shuffle.noising(x, x_len, 0)
            for i in range(len(x_len)):
                for j in range(x_len[i]):
                    self.assertEqual(x[j][i], x_noised[j][i])
            self.assertEqual(x_len[0], l_noised[0])

            x_noised, l_noised = word_shuffle.noising(x, x_len, 3)
            # Expect the second example has the last three tokens shuffled
            # 6, 7, 8, 9 => 6, 8, 9, 7, where (8, 9) is a word
            for i in range(x_len[0]):
                self.assertEqual(x[i][0], x_noised[i][0])
            shuffle_map = {0: 0, 1: 3, 2: 1, 3: 2}
            for k, v in shuffle_map.items():
                self.assertEqual(x[k][1], x_noised[v][1])
            self.assertEqual(x_len[0], l_noised[0])
            self.assertEqual(x_len[1], l_noised[1])


if __name__ == '__main__':
    unittest.main()
