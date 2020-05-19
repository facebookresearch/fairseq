import unittest

import tests.utils as test_utils
import torch

from fairseq.modules.inference_dropout_module import InferenceDropoutModule


class TestLayer(InferenceDropoutModule):
    pass


class TestInferenceDropout(unittest.TestCase):

    def setUp(self):
        self.tgt_dict, self.w1, self.w2, src_tokens, src_lengths, self.model = (
            test_utils.sequence_generator_setup()
        )
        self.model.encoder.layers = torch.nn.ModuleList([])
        self.model.encoder.layers.extend([
            TestLayer()
        ])

    def test_sets_retain_dropout_attribute(self):
        self.model.set_inference_dropout()
        assert self.model.retain_dropout
        assert self.model.encoder.retain_dropout
        assert self.model.encoder.layers[0].retain_dropout
        assert self.model.decoder.retain_dropout

    def test_sets_retain_dropout_attribute_specific_modules(self):
        self.model.set_inference_dropout(module_names=['TestEncoder'])
        assert self.model.retain_dropout
        assert self.model.encoder.retain_dropout
        assert not self.model.encoder.layers[0].retain_dropout
        assert not self.model.decoder.retain_dropout
