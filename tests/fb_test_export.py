#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from fairseq.models.fb_aan_transformer import AANTransformerModel
from tests.test_export import _test_save_and_load, get_dummy_task_and_parser


class TestExportModelsFB(unittest.TestCase):
    @unittest.skipIf(
        torch.__version__ < "1.6.0", "Targeting OSS scriptability for the 1.6 release"
    )
    def test_export_aan_transformer(self):
        task, parser = get_dummy_task_and_parser()
        AANTransformerModel.add_args(parser)
        args = parser.parse_args([])
        model = AANTransformerModel.build_model(args, task)
        scripted = torch.jit.script(model)
        _test_save_and_load(scripted)


if __name__ == "__main__":
    unittest.main()
