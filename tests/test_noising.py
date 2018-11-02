# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import unittest

import tests.utils as test_utils
import torch
from fairseq import utils
from fairseq.data import (
    AppendEosDataset,
    Dictionary,
    LanguagePairDataset,
    data_utils,
    noising,
)


class TestDataNoising(unittest.TestCase):
    def _get_test_data(self, append_eos=True, bpe=True):
        vocab = Dictionary()
        if bpe:
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
        else:
            vocab.add_symbol("hello")
            vocab.add_symbol("how")
            vocab.add_symbol("are")
            vocab.add_symbol("you")
            vocab.add_symbol("new")
            vocab.add_symbol("york")
            src_tokens = [
                ["hello", "new", "york", "you"],
                ["how", "are", "you", "new", "york"],
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
        """Asserts last token of every sentence in x is EOS """
        for i in range(len(x_len)):
            self.assertEqual(
                x[x_len[i] - 1][i],
                eos,
                (
                    "Expected eos (token id {eos}) at the end of sentence {i} but "
                    "got {other} instead"
                ).format(i=i, eos=eos, other=x[i][-1]),
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

    def assert_nonbpe_shuffle_with_distance_3(self, x, x_noised, x_len, l_noised):
        """
        Applies word shuffle with max_shuffle_distance = 3 and asserts that the
        shuffling result is as expected. If test data changes, update this func
        """
        # Expect the first example has the last two tokens shuffled
        # Expect the secon example has the second and third tokens shuffled
        shuffle_map = {0: 0, 1: 1, 2: 3, 3: 2}
        for k, v in shuffle_map.items():
            self.assertEqual(x[k][0], x_noised[v][0])
        shuffle_map = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
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

    def test_word_shuffle_with_eos_nonbpe(self):
        vocab, x, x_len = self._get_test_data(append_eos=True, bpe=False)

        with data_utils.numpy_seed(1234):
            word_shuffle = noising.WordShuffle(vocab, bpe_cont_marker=None)

            x_noised, l_noised = word_shuffle.noising(x, x_len, 0)
            self.assert_no_shuffle_with_0_distance(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

            x_noised, l_noised = word_shuffle.noising(x, x_len, 3)
            self.assert_nonbpe_shuffle_with_distance_3(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def assert_no_eos_at_end(self, x, x_len, eos):
        """Asserts that the last token of each sentence in x is not EOS """
        for i in range(len(x_len)):
            self.assertNotEqual(
                x[x_len[i] - 1][i],
                eos,
                "Expected no eos (token id {eos}) at the end of sentence {i}.".format(
                    eos=eos, i=i
                ),
            )

    def test_word_dropout_without_eos(self):
        """Same result as word dropout with eos except no EOS at end"""
        vocab, x, x_len = self._get_test_data(append_eos=False)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2)
            self.assert_word_dropout_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised
            )
            self.assert_no_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def test_word_blank_without_eos(self):
        """Same result as word blank with eos except no EOS at end"""
        vocab, x, x_len = self._get_test_data(append_eos=False)

        with data_utils.numpy_seed(1234):
            noising_gen = noising.WordDropout(vocab)
            x_noised, l_noised = noising_gen.noising(x, x_len, 0.2, vocab.unk())
            self.assert_word_blanking_correct(
                x=x, x_noised=x_noised, x_len=x_len, l_noised=l_noised, unk=vocab.unk()
            )
            self.assert_no_eos_at_end(x=x_noised, x_len=l_noised, eos=vocab.eos())

    def test_word_shuffle_without_eos(self):
        """Same result as word shuffle with eos except no EOS at end"""
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

    def _get_noising_dataset_batch(
        self, src_tokens_no_pad, src_dict, use_append_eos_dataset=False
    ):
        """
        Constructs a NoisingDataset and the corresponding
        LanguagePairDataset(NoisingDataset(src), src). If we set
        use_append_eos_dataset to True, wrap the source dataset in
        AppendEosDataset to append EOS to the clean source when using it as the
        target. In practice, we should use AppendEosDataset because our models
        usually have source without EOS but target with EOS.
        """
        src_dataset = test_utils.TestDataset(data=src_tokens_no_pad)

        noising_dataset = noising.NoisingDataset(
            src_dataset=src_dataset,
            src_dict=src_dict,
            seed=1234,
            max_word_shuffle_distance=3,
            word_dropout_prob=0.2,
            word_blanking_prob=0.2,
            noising_class=noising.UnsupervisedMTNoising,
        )
        tgt = src_dataset
        if use_append_eos_dataset:
            tgt = AppendEosDataset(src_dataset, src_dict.eos())
        language_pair_dataset = LanguagePairDataset(
            src=noising_dataset, tgt=tgt, src_sizes=None, src_dict=src_dict
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=language_pair_dataset,
            batch_size=2,
            collate_fn=language_pair_dataset.collater,
        )
        denoising_batch_result = next(iter(dataloader))
        return denoising_batch_result

    def test_noising_dataset_with_eos(self):
        src_dict, src_tokens, _ = self._get_test_data(append_eos=True)

        # Format data for src_dataset
        src_tokens = torch.t(src_tokens)
        src_tokens_no_pad = []
        for src_sentence in src_tokens:
            src_tokens_no_pad.append(
                utils.strip_pad(tensor=src_sentence, pad=src_dict.pad())
            )
        denoising_batch_result = self._get_noising_dataset_batch(
            src_tokens_no_pad=src_tokens_no_pad, src_dict=src_dict
        )

        eos, pad = src_dict.eos(), src_dict.pad()

        # Generated noisy source as source
        expected_src = torch.LongTensor(
            [[4, 5, 10, 11, 8, 12, 13, eos], [pad, pad, pad, 6, 8, 9, 7, eos]]
        )
        # Original clean source as target (right-padded)
        expected_tgt = torch.LongTensor(
            [[4, 5, 10, 11, 8, 12, 13, eos], [6, 7, 8, 9, eos, pad, pad, pad]]
        )
        generated_src = denoising_batch_result["net_input"]["src_tokens"]
        tgt_tokens = denoising_batch_result["target"]

        self.assertTensorEqual(expected_src, generated_src)
        self.assertTensorEqual(expected_tgt, tgt_tokens)

    def test_noising_dataset_without_eos(self):
        """
        Similar to test noising dataset with eos except that we have to set
        use_append_eos_dataset=True so that we wrap the source dataset in the
        AppendEosDataset when using it as the target in LanguagePairDataset.
        """

        src_dict, src_tokens, _ = self._get_test_data(append_eos=False)

        # Format data for src_dataset
        src_tokens = torch.t(src_tokens)
        src_tokens_no_pad = []
        for src_sentence in src_tokens:
            src_tokens_no_pad.append(
                utils.strip_pad(tensor=src_sentence, pad=src_dict.pad())
            )
        denoising_batch_result = self._get_noising_dataset_batch(
            src_tokens_no_pad=src_tokens_no_pad,
            src_dict=src_dict,
            use_append_eos_dataset=True,
        )

        eos, pad = src_dict.eos(), src_dict.pad()

        # Generated noisy source as source
        expected_src = torch.LongTensor(
            [[4, 5, 10, 11, 8, 12, 13], [pad, pad, pad, 6, 8, 9, 7]]
        )
        # Original clean source as target (right-padded)
        expected_tgt = torch.LongTensor(
            [[4, 5, 10, 11, 8, 12, 13, eos], [6, 7, 8, 9, eos, pad, pad, pad]]
        )

        generated_src = denoising_batch_result["net_input"]["src_tokens"]
        tgt_tokens = denoising_batch_result["target"]

        self.assertTensorEqual(expected_src, generated_src)
        self.assertTensorEqual(expected_tgt, tgt_tokens)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


if __name__ == "__main__":
    unittest.main()
