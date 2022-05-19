# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import typing as tp
import unittest
from tempfile import TemporaryDirectory

from fairseq.binarizer import BinarizeSummary, FileBinarizer, VocabularyDatasetBinarizer
from fairseq.data import Dictionary, indexed_dataset
from tests.utils import make_data, sizes


def build_vocab(data: tp.List[tp.List[str]]) -> Dictionary:
    d = Dictionary()
    for s in data:
        for token in s:
            d.add_symbol(token)
    d.finalize()
    return d


class TestBinarizer(unittest.TestCase):
    def compare_ds_data(self, summary, data, prefix, impl, vocab):
        self.assertEqual(summary.num_seq, len(data))
        self.assertEqual(summary.num_tok, sum([len(s) for s in data]))

        dataset = indexed_dataset.make_dataset(prefix, impl)

        self.assertEqual(len(dataset), len(data))
        decoded = [vocab.string(dataset[i]).split() for i in range(0, len(dataset))]

        self.assertEqual(decoded, data)
        data_sizes = [i.item() for i in dataset.sizes]
        self.assertEqual(data_sizes, sizes(data))

    def test_can_binarize_line(self):
        data = make_data(length=1)
        vocab = build_vocab(data)

        binarizer = VocabularyDatasetBinarizer(
            vocab,
        )

        sentence = data[0]
        summary = BinarizeSummary()

        tensor = binarizer.binarize_line(
            " ".join(sentence),
            summary,
        )

        self.assertEqual(len(tensor), len(sentence) + 1)

        self.assertEqual(summary.num_tok, len(sentence) + 1)
        self.assertEqual(summary.num_seq, 1)

    def test_can_binarize_file_chunk(self):
        # test without multiprocess logic
        with TemporaryDirectory() as dirname:
            raw_file = os.path.join(dirname, "raw1")
            prefix = os.path.join(dirname, "test1")
            impl = "mmap"

            data = make_data(out_file=raw_file)
            vocab = build_vocab(data)

            binarizer = VocabularyDatasetBinarizer(
                vocab,
                append_eos=False,
            )

            summary = FileBinarizer._binarize_chunk_and_finalize(
                binarizer,
                raw_file,
                offset_start=0,
                offset_end=-1,
                output_prefix=prefix,
                dataset_impl=impl,
                vocab_size=len(vocab),
            )

            self.compare_ds_data(summary, data, prefix, impl, vocab)

    def test_can_multiprocess(self):
        with TemporaryDirectory() as dirname:
            raw_file = os.path.join(dirname, "raw1")
            prefix = os.path.join(dirname, "test1")
            impl = "mmap"
            data = make_data(out_file=raw_file)
            vocab = build_vocab(data)
            binarizer = VocabularyDatasetBinarizer(
                vocab,
                append_eos=False,
            )
            # with one worker
            summary = FileBinarizer.multiprocess_dataset(
                raw_file,
                impl,
                binarizer,
                output_prefix=prefix,
                vocab_size=len(vocab),
                num_workers=1,
            )

            self.compare_ds_data(summary, data, prefix, impl, vocab)

            # with multiple worker
            prefix_multi = os.path.join(dirname, "test2")
            summary = FileBinarizer.multiprocess_dataset(
                raw_file,
                impl,
                binarizer,
                output_prefix=prefix_multi,
                vocab_size=len(vocab),
                num_workers=3,
            )

            self.compare_ds_data(summary, data, prefix_multi, impl, vocab)
