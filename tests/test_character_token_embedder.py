# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unittest

from fairseq.data import Dictionary
from fairseq.modules import CharacterTokenEmbedder


class TestCharacterTokenEmbedder(unittest.TestCase):
    def test_character_token_embedder(self):
        vocab = Dictionary()
        vocab.add_symbol('hello')
        vocab.add_symbol('there')

        embedder = CharacterTokenEmbedder(vocab, [(2, 16), (4, 32), (8, 64), (16, 2)], 64, 5, 2)

        test_sents = [['hello', 'unk', 'there'], ['there'], ['hello', 'there']]
        max_len = max(len(s) for s in test_sents)
        input = torch.LongTensor(len(test_sents), max_len + 2).fill_(vocab.pad())
        for i in range(len(test_sents)):
            input[i][0] = vocab.eos()
            for j in range(len(test_sents[i])):
                input[i][j + 1] = vocab.index(test_sents[i][j])
            input[i][j + 2] = vocab.eos()
        embs = embedder(input)

        assert embs.size() == (len(test_sents), max_len + 2, 5)
        self.assertAlmostEqual(embs[0][0], embs[1][0])
        self.assertAlmostEqual(embs[0][0], embs[0][-1])
        self.assertAlmostEqual(embs[0][1], embs[2][1])
        self.assertAlmostEqual(embs[0][3], embs[1][1])

        embs.sum().backward()
        assert embedder.char_embeddings.weight.grad is not None

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
