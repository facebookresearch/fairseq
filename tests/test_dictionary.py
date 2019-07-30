# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch

from fairseq.data import Dictionary


class TestDictionary(unittest.TestCase):

    def test_finalize(self):
        txt = [
            'A B C D',
            'B C D',
            'C D',
            'D',
        ]
        ref_ids1 = list(map(torch.IntTensor, [
            [4, 5, 6, 7, 2],
            [5, 6, 7, 2],
            [6, 7, 2],
            [7, 2],
        ]))
        ref_ids2 = list(map(torch.IntTensor, [
            [7, 6, 5, 4, 2],
            [6, 5, 4, 2],
            [5, 4, 2],
            [4, 2],
        ]))

        # build dictionary
        d = Dictionary()
        for line in txt:
            d.encode_line(line, add_if_not_exist=True)

        def get_ids(dictionary):
            ids = []
            for line in txt:
                ids.append(dictionary.encode_line(line, add_if_not_exist=False))
            return ids

        def assertMatch(ids, ref_ids):
            for toks, ref_toks in zip(ids, ref_ids):
                self.assertEqual(toks.size(), ref_toks.size())
                self.assertEqual(0, (toks != ref_toks).sum().item())

        ids = get_ids(d)
        assertMatch(ids, ref_ids1)

        # check finalized dictionary
        d.finalize()
        finalized_ids = get_ids(d)
        assertMatch(finalized_ids, ref_ids2)

        # write to disk and reload
        with tempfile.NamedTemporaryFile(mode='w') as tmp_dict:
            d.save(tmp_dict.name)
            d = Dictionary.load(tmp_dict.name)
            reload_ids = get_ids(d)
            assertMatch(reload_ids, ref_ids2)
            assertMatch(finalized_ids, reload_ids)


if __name__ == '__main__':
    unittest.main()
