# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from fairseq.models.transformer import TransformerModel
from tests.test_sequence_generator import get_dummy_task_and_parser


class TestInferenceDropout(unittest.TestCase):
    def setUp(self):
        self.task, self.parser = get_dummy_task_and_parser()
        TransformerModel.add_args(self.parser)
        self.args = self.parser.parse_args([])
        self.args.encoder_layers = 2
        self.args.decoder_layers = 1
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_sets_inference_dropout_to_true(self):
        self.args.retain_dropout = True
        self.transformer_model = TransformerModel.build_model(self.args, self.task)
        self.transformer_model.prepare_for_inference_(self.args)
        assert self.transformer_model.encoder.dropout_module.apply_during_inference
        assert self.transformer_model.decoder.dropout_module.apply_during_inference
        for layer in self.transformer_model.encoder.layers:
            assert layer.dropout_module.apply_during_inference

    def test_inference_dropout_false_by_default(self):
        self.transformer_model = TransformerModel.build_model(self.args, self.task)
        self.transformer_model.prepare_for_inference_(self.args)
        assert not self.transformer_model.encoder.dropout_module.apply_during_inference
        assert not self.transformer_model.decoder.dropout_module.apply_during_inference
        for layer in self.transformer_model.encoder.layers:
            assert not layer.dropout_module.apply_during_inference
        for layer in self.transformer_model.decoder.layers:
            assert not layer.dropout_module.apply_during_inference

    def test_applies_training_mode(self):
        self.transformer_model = TransformerModel.build_model(self.args, self.task)
        assert self.transformer_model.encoder.dropout_module.training
        for layer in self.transformer_model.encoder.layers:
            assert layer.dropout_module.training

        self.transformer_model.eval()
        assert not self.transformer_model.decoder.dropout_module.training
        for layer in self.transformer_model.encoder.layers:
            assert not layer.dropout_module.training

    def test_retain_modules(self):
        self.args.retain_dropout = True
        self.args.retain_dropout_modules = [
            "TransformerEncoder",
            "TransformerEncoderLayer",
        ]
        self.transformer_model = TransformerModel.build_model(self.args, self.task)
        self.transformer_model.prepare_for_inference_(self.args)
        assert self.transformer_model.encoder.dropout_module.apply_during_inference
        assert not self.transformer_model.decoder.dropout_module.apply_during_inference
        for layer in self.transformer_model.decoder.layers:
            assert not layer.dropout_module.apply_during_inference
