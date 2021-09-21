# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from argparse import ArgumentParser
from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass


@dataclass
class A(FairseqDataclass):
    data: str = field(default="test", metadata={"help": "the data input"})
    num_layers: int = field(default=200, metadata={"help": "more layers is better?"})


@dataclass
class B(FairseqDataclass):
    bar: A = field(default=A())
    foo: int = field(default=0, metadata={"help": "not a bar"})


@dataclass
class D(FairseqDataclass):
    arch: A = field(default=A())
    foo: int = field(default=0, metadata={"help": "not a bar"})


@dataclass
class C(FairseqDataclass):
    data: str = field(default="test", metadata={"help": "root level data input"})
    encoder: D = field(default=D())
    decoder: A = field(default=A())
    lr: int = field(default=0, metadata={"help": "learning rate"})


class TestDataclassUtils(unittest.TestCase):
    def test_argparse_convert_basic(self):
        parser = ArgumentParser()
        gen_parser_from_dataclass(parser, A(), True)
        args = parser.parse_args(["--num-layers", '10', "the/data/path"])
        self.assertEqual(args.num_layers, 10)
        self.assertEqual(args.data, "the/data/path")

    def test_argparse_recursive(self):
        parser = ArgumentParser()
        gen_parser_from_dataclass(parser, B(), True)
        args = parser.parse_args(["--num-layers", "10", "--foo", "10", "the/data/path"])
        self.assertEqual(args.num_layers, 10)
        self.assertEqual(args.foo, 10)
        self.assertEqual(args.data, "the/data/path")

    def test_argparse_recursive_prefixing(self):
        self.maxDiff = None
        parser = ArgumentParser()
        gen_parser_from_dataclass(parser, C(), True, "")
        args = parser.parse_args(
            [
                "--encoder-arch-data",
                "ENCODER_ARCH_DATA",
                "--encoder-arch-num-layers",
                "10",
                "--encoder-foo",
                "10",
                "--decoder-data",
                "DECODER_DATA",
                "--decoder-num-layers",
                "10",
                "--lr",
                "10",
                "the/data/path",
            ]
        )
        self.assertEqual(args.encoder_arch_data, "ENCODER_ARCH_DATA")
        self.assertEqual(args.encoder_arch_num_layers, 10)
        self.assertEqual(args.encoder_foo, 10)
        self.assertEqual(args.decoder_data, "DECODER_DATA")
        self.assertEqual(args.decoder_num_layers, 10)
        self.assertEqual(args.lr, 10)
        self.assertEqual(args.data, "the/data/path")


if __name__ == "__main__":
    unittest.main()
