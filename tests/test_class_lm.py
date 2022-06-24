import unittest

import numpy as np
from examples.speech_text_joint_to_text.models.class_lm import ClassBasedLanguageModel
from fairseq.data.indexed_dataset import IndexedRawTextDataset
from fairseq.data.pad_dataset import PadDataset
from fairseq.models.fairseq_decoder import FairseqDecoder
import torch

from fairseq.data.dictionary import Dictionary
from fairseq.models.fairseq_model import FairseqLanguageModel


class TestLMDecoder(FairseqDecoder):
    def __init__(self, dictionary: Dictionary):
        super().__init__(dictionary)
        self.tgt_len = len(dictionary)
        self.onnx_trace = False

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        tmp = torch.ones((prev_output_tokens.shape[0], prev_output_tokens.shape[1], self.tgt_len))
        tmp[:, :, 0] *= 10.
        return tmp


class TestRawDataset(IndexedRawTextDataset):
    def __init__(self, lines, dictionary, append_eos=True):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.lines = lines
        for line in self.lines:
            tokens = dictionary.encode_line(
                line,
                add_if_not_exist=False,
                append_eos=append_eos,
            ).long()
            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)
        self.size = len(self.tokens_list)


class ClassLMTestCase(unittest.TestCase):

    def setUp(self):
        self.dictionary = Dictionary(extra_special_symbols=[
            "<A>", "</A>", "<B>", "</B>", "<C>", "</C>"])
        src_lines = [
            "I like <A> quokkas </A>",
            "I like <B> tortoises </B> and <A> quokkas </A>",
            "I like elephants",
            "I like <B> tortoises </B> and <A> quokkas </A> and elephants"]
        for l in src_lines:
            self.dictionary.encode_line(l)
        self.__underlying_lm = FairseqLanguageModel(TestLMDecoder(self.dictionary))
        self.lm = ClassBasedLanguageModel(self.__underlying_lm, self.dictionary, ["A", "B", "C"])
        self.ds = PadDataset(
            TestRawDataset(src_lines, self.dictionary),
            pad_idx=self.dictionary.pad(),
            left_pad=False,
        )

    def test_basic(self):
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2], self.ds[3]])
        outs = self.lm(samples)
        out_probs = self.lm.get_normalized_probs(outs, log_probs=True)
        self.assertEqual(list(out_probs.shape), [4, 1, len(self.dictionary)])
        equal_probs_vector = torch.log_softmax(torch.tensor([0.] * len(self.dictionary)), dim=-1).tolist()
        self.assertEqual([
            [equal_probs_vector], [equal_probs_vector], [equal_probs_vector], [equal_probs_vector]],
             out_probs.tolist())

    def __probs_up_to_idx(self, idx):
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2], self.ds[3]])
        outs = self.lm(samples[:, :idx])
        out_probs = self.lm.get_normalized_probs(outs, log_probs=True)
        self.assertEqual(list(out_probs.shape), [4, 1, len(self.dictionary)])
        return out_probs

    def test_boundary_detection(self):
        equal_probs_vector = torch.log_softmax(torch.tensor([0.] * len(self.dictionary)), dim=-1).tolist()
        in_tag_probs = torch.log_softmax(
            torch.cat([torch.tensor([10.]), torch.tensor([1.] * (len(self.dictionary) - 1))]), dim=-1).tolist()
        for idx in [1, 2, 5, 6, 9, 10, 11]:
            probs = self.__probs_up_to_idx(idx)
            self.assertEqual([
                [equal_probs_vector], [equal_probs_vector], [equal_probs_vector], [equal_probs_vector]],
                probs.tolist())
        for idx in [3, 4]:
            probs = self.__probs_up_to_idx(idx)
            self.assertEqual([
                    [in_tag_probs], [in_tag_probs], [equal_probs_vector], [in_tag_probs]],
                    probs.tolist())
        for idx in [7, 8]:
            probs = self.__probs_up_to_idx(idx)
            self.assertEqual([
                    [equal_probs_vector], [in_tag_probs], [equal_probs_vector], [in_tag_probs]],
                    probs.tolist())


if __name__ == '__main__':
    unittest.main()
