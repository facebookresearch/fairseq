# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import tempfile
import unittest

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.lstm import LSTMModel
from fairseq.tasks.fairseq_task import LegacyFairseqTask


DEFAULT_TEST_VOCAB_SIZE = 100


class DummyTask(LegacyFairseqTask):
    def __init__(self, args):
        super().__init__(args)
        self.dictionary = get_dummy_dictionary()
        if getattr(self.args, "ctc", False):
            self.dictionary.add_symbol("<ctc_blank>")
        self.src_dict = self.dictionary
        self.tgt_dict = self.dictionary

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.dictionary


def get_dummy_dictionary(vocab_size=DEFAULT_TEST_VOCAB_SIZE):
    dummy_dict = Dictionary()
    # add dummy symbol to satisfy vocab size
    for id, _ in enumerate(range(vocab_size)):
        dummy_dict.add_symbol("{}".format(id), 1000)
    return dummy_dict


def get_dummy_task_and_parser():
    """
    to build a fariseq model, we need some dummy parse and task. This function
    is used to create dummy task and parser to faciliate model/criterion test

    Note: we use FbSpeechRecognitionTask as the dummy task. You may want
    to use other task by providing another function
    """
    parser = argparse.ArgumentParser(
        description="test_dummy_s2s_task", argument_default=argparse.SUPPRESS
    )
    DummyTask.add_args(parser)
    args = parser.parse_args([])
    task = DummyTask.setup_task(args)
    return task, parser


class TestJitLSTMModel(unittest.TestCase):
    def _test_save_and_load(self, scripted_module):
        with tempfile.NamedTemporaryFile() as f:
            scripted_module.save(f.name)
            torch.jit.load(f.name)

    def assertTensorEqual(self, t1, t2):
        t1 = t1[~torch.isnan(t1)]  # can cause size mismatch errors if there are NaNs
        t2 = t2[~torch.isnan(t2)]
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)

    def test_jit_and_export_lstm(self):
        task, parser = get_dummy_task_and_parser()
        LSTMModel.add_args(parser)
        args = parser.parse_args([])
        args.criterion = ""
        model = LSTMModel.build_model(args, task)
        scripted_model = torch.jit.script(model)
        self._test_save_and_load(scripted_model)

    def test_assert_jit_vs_nonjit_(self):
        task, parser = get_dummy_task_and_parser()
        LSTMModel.add_args(parser)
        args = parser.parse_args([])
        args.criterion = ""
        model = LSTMModel.build_model(args, task)
        model.eval()
        scripted_model = torch.jit.script(model)
        scripted_model.eval()
        idx = len(task.source_dictionary)
        iter = 100
        # Inject random input and check output
        seq_len_tensor = torch.randint(1, 10, (iter,))
        num_samples_tensor = torch.randint(1, 10, (iter,))
        for i in range(iter):
            seq_len = seq_len_tensor[i]
            num_samples = num_samples_tensor[i]
            src_token = (torch.randint(0, idx, (num_samples, seq_len)),)
            src_lengths = torch.randint(1, seq_len + 1, (num_samples,))
            src_lengths, _ = torch.sort(src_lengths, descending=True)
            # Force the first sample to have seq_len
            src_lengths[0] = seq_len
            prev_output_token = (torch.randint(0, idx, (num_samples, 1)),)
            result = model(src_token[0], src_lengths, prev_output_token[0], None)
            scripted_result = scripted_model(
                src_token[0], src_lengths, prev_output_token[0], None
            )
            self.assertTensorEqual(result[0], scripted_result[0])
            self.assertTensorEqual(result[1], scripted_result[1])


if __name__ == "__main__":
    unittest.main()
