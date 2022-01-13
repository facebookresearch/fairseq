# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import string
import tempfile
import unittest

import torch
from fairseq import tokenizer
from fairseq.data import Dictionary


class TestDictionary(unittest.TestCase):
    def test_finalize(self):
        txt = [
            "A B C D",
            "B C D",
            "C D",
            "D",
        ]
        ref_ids1 = list(
            map(
                torch.IntTensor,
                [
                    [4, 5, 6, 7, 2],
                    [5, 6, 7, 2],
                    [6, 7, 2],
                    [7, 2],
                ],
            )
        )
        ref_ids2 = list(
            map(
                torch.IntTensor,
                [
                    [7, 6, 5, 4, 2],
                    [6, 5, 4, 2],
                    [5, 4, 2],
                    [4, 2],
                ],
            )
        )

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
        with tempfile.NamedTemporaryFile(mode="w") as tmp_dict:
            d.save(tmp_dict.name)
            d = Dictionary.load(tmp_dict.name)
            reload_ids = get_ids(d)
            assertMatch(reload_ids, ref_ids2)
            assertMatch(finalized_ids, reload_ids)

    def test_overwrite(self):
        # for example, Camembert overwrites <unk>, <s> and </s>
        dict_file = io.StringIO(
            "<unk> 999 #fairseq:overwrite\n"
            "<s> 999 #fairseq:overwrite\n"
            "</s> 999 #fairseq:overwrite\n"
            ", 999\n"
            "▁de 999\n"
        )
        d = Dictionary()
        d.add_from_file(dict_file)
        self.assertEqual(d.index("<pad>"), 1)
        self.assertEqual(d.index("foo"), 3)
        self.assertEqual(d.index("<unk>"), 4)
        self.assertEqual(d.index("<s>"), 5)
        self.assertEqual(d.index("</s>"), 6)
        self.assertEqual(d.index(","), 7)
        self.assertEqual(d.index("▁de"), 8)

    def test_no_overwrite(self):
        # for example, Camembert overwrites <unk>, <s> and </s>
        dict_file = io.StringIO(
            "<unk> 999\n" "<s> 999\n" "</s> 999\n" ", 999\n" "▁de 999\n"
        )
        d = Dictionary()
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            d.add_from_file(dict_file)

    def test_space(self):
        # for example, character models treat space as a symbol
        dict_file = io.StringIO("  999\n" "a 999\n" "b 999\n")
        d = Dictionary()
        d.add_from_file(dict_file)
        self.assertEqual(d.index(" "), 4)
        self.assertEqual(d.index("a"), 5)
        self.assertEqual(d.index("b"), 6)

    def test_add_file_to_dict(self):
        counts = {}
        num_lines = 100
        per_line = 10
        with tempfile.TemporaryDirectory("test_sampling") as data_dir:
            filename = os.path.join(data_dir, "dummy.txt")
            with open(filename, "w", encoding="utf-8") as data:
                for c in string.ascii_letters:
                    line = f"{c} " * per_line
                    for _ in range(num_lines):
                        data.write(f"{line}\n")
                    counts[c] = per_line * num_lines
                    per_line += 5

            dict = Dictionary()
            Dictionary.add_file_to_dictionary(
                filename, dict, tokenizer.tokenize_line, 10
            )
            dict.finalize(threshold=0, nwords=-1, padding_factor=8)

            for c in string.ascii_letters:
                count = dict.get_count(dict.index(c))
                self.assertEqual(
                    counts[c], count, f"{c} count is {count} but should be {counts[c]}"
                )


if __name__ == "__main__":
    unittest.main()
