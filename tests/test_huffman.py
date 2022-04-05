# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
import unittest
from collections import Counter
from tempfile import NamedTemporaryFile, TemporaryDirectory

from fairseq.data import Dictionary, indexed_dataset
from fairseq.data.huffman import (
    HuffmanCodeBuilder,
    HuffmanCoder,
    HuffmanMMapIndexedDataset,
    HuffmanMMapIndexedDatasetBuilder,
)
from tests.utils import POPULATION, make_data, sizes


def make_counts(data: tp.List[tp.List[str]]) -> Counter:
    return Counter([symbol for sentence in data for symbol in sentence])


def make_code_builder(data: tp.List[tp.List[str]]) -> HuffmanCodeBuilder:
    builder = HuffmanCodeBuilder()
    for sentence in data:
        builder.add_symbols(*sentence)
    return builder


class TestCodeBuilder(unittest.TestCase):
    def test_code_builder_can_count(self):
        data = make_data()
        counts = make_counts(data)
        builder = make_code_builder(data)

        self.assertEqual(builder.symbols, counts)

    def test_code_builder_can_add(self):
        data = make_data()
        counts = make_counts(data)
        builder = make_code_builder(data)

        new_builder = builder + builder

        self.assertEqual(new_builder.symbols, counts + counts)

    def test_code_builder_can_io(self):
        data = make_data()
        builder = make_code_builder(data)

        with NamedTemporaryFile() as tmp_fp:
            builder.to_file(tmp_fp.name)
            other_builder = HuffmanCodeBuilder.from_file(tmp_fp.name)

            self.assertEqual(builder.symbols, other_builder.symbols)


class TestCoder(unittest.TestCase):
    def test_coder_can_io(self):
        data = make_data()
        builder = make_code_builder(data)
        coder = builder.build_code()

        with NamedTemporaryFile() as tmp_fp:
            coder.to_file(tmp_fp.name)
            other_coder = HuffmanCoder.from_file(tmp_fp.name)

            self.assertEqual(coder, other_coder)

    def test_coder_can_encode_decode(self):
        data = make_data()
        builder = make_code_builder(data)
        coder = builder.build_code()

        encoded = [coder.encode(sentence) for sentence in data]
        decoded = [[n.symbol for n in coder.decode(enc)] for enc in encoded]

        self.assertEqual(decoded, data)

        unseen_data = make_data()
        unseen_encoded = [coder.encode(sentence) for sentence in unseen_data]
        unseen_decoded = [
            [n.symbol for n in coder.decode(enc)] for enc in unseen_encoded
        ]
        self.assertEqual(unseen_decoded, unseen_data)


def build_dataset(prefix, data, coder):
    with HuffmanMMapIndexedDatasetBuilder(prefix, coder) as builder:
        for sentence in data:
            builder.add_item(sentence)


class TestHuffmanDataset(unittest.TestCase):
    def test_huffman_can_encode_decode(self):
        data = make_data()
        builder = make_code_builder(data)
        coder = builder.build_code()

        with TemporaryDirectory() as dirname:
            prefix = os.path.join(dirname, "test1")
            build_dataset(prefix, data, coder)
            dataset = HuffmanMMapIndexedDataset(prefix)

            self.assertEqual(len(dataset), len(data))
            decoded = [list(dataset.get_symbols(i)) for i in range(0, len(dataset))]

            self.assertEqual(decoded, data)
            data_sizes = [i.item() for i in dataset.sizes]
            self.assertEqual(data_sizes, sizes(data))

    def test_huffman_compresses(self):
        data = make_data()
        builder = make_code_builder(data)
        coder = builder.build_code()

        with TemporaryDirectory() as dirname:
            prefix = os.path.join(dirname, "huffman")
            build_dataset(prefix, data, coder)

            prefix_mmap = os.path.join(dirname, "mmap")
            mmap_builder = indexed_dataset.make_builder(
                indexed_dataset.data_file_path(prefix_mmap),
                "mmap",
                vocab_size=len(POPULATION),
            )
            dictionary = Dictionary()
            for c in POPULATION:
                dictionary.add_symbol(c)
            dictionary.finalize()
            for sentence in data:
                mmap_builder.add_item(dictionary.encode_line(" ".join(sentence)))
            mmap_builder.finalize(indexed_dataset.index_file_path(prefix_mmap))

            huff_size = os.stat(indexed_dataset.data_file_path(prefix)).st_size
            mmap_size = os.stat(indexed_dataset.data_file_path(prefix_mmap)).st_size
            self.assertLess(huff_size, mmap_size)

    def test_huffman_can_append(self):
        data1 = make_data()
        builder = make_code_builder(data1)
        coder = builder.build_code()

        with TemporaryDirectory() as dirname:
            prefix1 = os.path.join(dirname, "test1")
            build_dataset(prefix1, data1, coder)

            data2 = make_data()
            prefix2 = os.path.join(dirname, "test2")
            build_dataset(prefix2, data2, coder)

            prefix3 = os.path.join(dirname, "test3")

            with HuffmanMMapIndexedDatasetBuilder(prefix3, coder) as builder:
                builder.append(prefix1)
                builder.append(prefix2)

            dataset = HuffmanMMapIndexedDataset(prefix3)

            self.assertEqual(len(dataset), len(data1) + len(data2))

            decoded1 = [list(dataset.get_symbols(i)) for i in range(0, len(data1))]
            self.assertEqual(decoded1, data1)

            decoded2 = [
                list(dataset.get_symbols(i)) for i in range(len(data1), len(dataset))
            ]
            self.assertEqual(decoded2, data2)

            data_sizes = [i.item() for i in dataset.sizes]
            self.assertEqual(data_sizes[: len(data1)], sizes(data1))
            self.assertEqual(data_sizes[len(data1) : len(dataset)], sizes(data2))


if __name__ == "__main__":
    unittest.main()
