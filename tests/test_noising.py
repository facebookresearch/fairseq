# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import unittest

from fairseq.data import Dictionary, data_utils, noising


class TestDataNoising(unittest.TestCase):
    def _get_test_data(self, append_eos=True):
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
        # If we have to append EOS, we include EOS in counting src length
        if append_eos:
            src_len = [length + 1 for length in src_len]

        x = torch.LongTensor(len(src_tokens), max(src_len)).fill_(vocab.pad())
        for i in range(len(src_tokens)):
            for j in range(len(src_tokens[i])):
                x[i][j] = vocab.index(src_tokens[i][j])
            if append_eos:
                x[i][j + 1] = vocab.eos()

        x = x.transpose(1, 0)
        return vocab, x, torch.LongTensor(src_len)

    def assert_eos_at_end(self, x, x_len, eos):
        """ Asserts last token of every sentence in x is EOS """
        for i in range(len(x_len)):
            self.assertEqual(
                x[x_len[i]-1][i],
                eos,
                f"Expected eos (token id {eos}) at the end of sentence {i} but "
                f"got {x[i][-1]} instead"
            )

    def assert_word_dropout_correct(self, x, x_noised, x_len, l_noised):
        # Expect only the first word (2 bpe tokens) of the first example
        # was dropped out
        self.assertEqual(x_len[0] - 2, l_noised[0])
        for i in range(l_noised[0]):
            self.assertEqual(x_noised[i][0], x[i + 2][0])

    def test_word_dropout_with_eos(self):
        vocab, x, x_len = self._get_test_data(append_eos=True)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2)
            self.assert_word_dropout_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def assert_word_blanking_correct(self, x, x_noised, x_len, l_noised, unk):
        # Expect only the first word (2 bpe tokens) of the first example
        # was blanked out
        self.assertEqual(x_len[0], l_noised[0])
        for i in range(l_noised[0]):
            if i < 2:
                self.assertEqual(x_noised[i][0], unk)
            else:
                self.assertEqual(x_noised[i][0], x[i][0])

    def test_word_blank_with_eos(self):
        vocab, x, x_len = self._get_test_data(append_eos=True)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2, vocab.unk())
            self.assert_word_blanking_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised, unk=vocab.unk()
            )
            self.assert_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def assert_no_shuffle_with_0_distance(self, x, x_noised, x_len, l_noised):
        """
        Applies word shuffle with 0 max_shuffle_distance and asserts that no
        shuffling happened
        """
        for i in range(len(x_len)):
            for j in range(x_len[i]):
                self.assertEqual(x[j][i], x_noised[j][i])
        self.assertEqual(x_len[0], l_noised[0])

    def assert_word_shuffle_with_distance_3(self, x, x_noised, x_len, l_noised):
        """
        Applies word shuffle with max_shuffle_distance = 3 and asserts that the
        shuffling result is as expected. If test data changes, update this func
        """
        # Expect the second example has the last three tokens shuffled
        # 6, 7, 8, 9 => 6, 8, 9, 7, where (8, 9) is a word
        for i in range(x_len[0]):
            self.assertEqual(x[i][0], x_noised[i][0])
        shuffle_map = {0: 0, 1: 3, 2: 1, 3: 2}
        for k, v in shuffle_map.items():
            self.assertEqual(x[k][1], x_noised[v][1])
        self.assertEqual(x_len[0], l_noised[0])
        self.assertEqual(x_len[1], l_noised[1])

    def test_word_shuffle_with_eos(self):
        vocab, x, x_len = self._get_test_data(append_eos=True)

        with data_utils.numpy_seed(1234):
            word_shuffle = noising.WordShuffle(vocab)

            x_noised, l_noised = word_shuffle.noising(x, x_len, 0)
            self.assert_no_shuffle_with_0_distance(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

            x_noised, l_noised = word_shuffle.noising(x, x_len, 3)
            self.assert_word_shuffle_with_distance_3(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def assert_no_eos_at_end(self, x, x_len, eos):
        """ Asserts that the last token of each sentence in x is not EOS """
        for i in range(len(x_len)):
            self.assertNotEqual(
                x[x_len[i]-1][i],
                eos,
                f"Expected no eos (token id {eos}) at the end of sentence {i}."
            )

    def test_word_dropout_without_eos(self):
        """ Same result as word dropout with eos except no EOS at end"""
        vocab, x, x_len = self._get_test_data(append_eos=False)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2)
            self.assert_word_dropout_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_no_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def test_word_blank_without_eos(self):
        """ Same result as word blank with eos except no EOS at end"""
        vocab, x, x_len = self._get_test_data(append_eos=False)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2, vocab.unk())
            self.assert_word_blanking_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised, unk=vocab.unk()
            )
            self.assert_no_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def test_word_shuffle_without_eos(self):
        """ Same result as word shuffle with eos except no EOS at end """
        vocab, x, x_len = self._get_test_data(append_eos=False)

        with data_utils.numpy_seed(1234):
            word_shuffle = noising.WordShuffle(vocab)

            x_noised, l_noised = word_shuffle.noising(x, x_len, 0)
            self.assert_no_shuffle_with_0_distance(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_no_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

            x_noised, l_noised = word_shuffle.noising(x, x_len, 3)
            self.assert_word_shuffle_with_distance_3(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_no_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())


if __name__ == '__main__':
    unittest.main()
