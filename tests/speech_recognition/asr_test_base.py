#!/usr/bin/env python3

import argparse
import os
import unittest
from inspect import currentframe, getframeinfo

import numpy as np
import torch
from examples.speech_recognition.data.data_utils import lengths_to_encoder_padding_mask
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.dictionary import Dictionary
from fairseq.models import (
    BaseFairseqModel,
    FairseqDecoder,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqEncoderModel,
    FairseqModel,
)
from fairseq.tasks.fairseq_task import LegacyFairseqTask


DEFAULT_TEST_VOCAB_SIZE = 100


# ///////////////////////////////////////////////////////////////////////////
# utility function to setup dummy dict/task/input
# ///////////////////////////////////////////////////////////////////////////


def get_dummy_dictionary(vocab_size=DEFAULT_TEST_VOCAB_SIZE):
    dummy_dict = Dictionary()
    # add dummy symbol to satisfy vocab size
    for id, _ in enumerate(range(vocab_size)):
        dummy_dict.add_symbol("{}".format(id), 1000)
    return dummy_dict


class DummyTask(LegacyFairseqTask):
    def __init__(self, args):
        super().__init__(args)
        self.dictionary = get_dummy_dictionary()
        if getattr(self.args, "ctc", False):
            self.dictionary.add_symbol("<ctc_blank>")
        self.tgt_dict = self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


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


def get_dummy_input(T=100, D=80, B=5, K=100):
    forward_input = {}
    # T max sequence length
    # D feature vector dimension
    # B batch size
    # K target dimension size
    feature = torch.randn(B, T, D)
    # this (B, T, D) layout is just a convention, you can override it by
    # write your own _prepare_forward_input function
    src_lengths = torch.from_numpy(
        np.random.randint(low=1, high=T, size=B, dtype=np.int64)
    )
    src_lengths[0] = T  # make sure the maximum length matches
    prev_output_tokens = []
    for b in range(B):
        token_length = np.random.randint(low=1, high=src_lengths[b].item() + 1)
        tokens = np.random.randint(low=0, high=K, size=token_length, dtype=np.int64)
        prev_output_tokens.append(torch.from_numpy(tokens))

    prev_output_tokens = fairseq_data_utils.collate_tokens(
        prev_output_tokens,
        pad_idx=1,
        eos_idx=2,
        left_pad=False,
        move_eos_to_beginning=False,
    )
    src_lengths, sorted_order = src_lengths.sort(descending=True)
    forward_input["src_tokens"] = feature.index_select(0, sorted_order)
    forward_input["src_lengths"] = src_lengths
    forward_input["prev_output_tokens"] = prev_output_tokens

    return forward_input


def get_dummy_encoder_output(encoder_out_shape=(100, 80, 5)):
    """
    This only provides an example to generate dummy encoder output
    """
    (T, B, D) = encoder_out_shape
    encoder_out = {}

    encoder_out["encoder_out"] = torch.from_numpy(
        np.random.randn(*encoder_out_shape).astype(np.float32)
    )
    seq_lengths = torch.from_numpy(np.random.randint(low=1, high=T, size=B))
    # some dummy mask
    encoder_out["encoder_padding_mask"] = torch.arange(T).view(1, T).expand(
        B, -1
    ) >= seq_lengths.view(B, 1).expand(-1, T)
    encoder_out["encoder_padding_mask"].t_()

    # encoer_padding_mask is (T, B) tensor, with (t, b)-th element indicate
    # whether encoder_out[t, b] is valid (=0) or not (=1)
    return encoder_out


def _current_postion_info():
    cf = currentframe()
    frameinfo = " (at {}:{})".format(
        os.path.basename(getframeinfo(cf).filename), cf.f_back.f_lineno
    )
    return frameinfo


def check_encoder_output(encoder_output, batch_size=None):
    """we expect encoder_output to be a dict with the following
    key/value pairs:
    - encoder_out: a Torch.Tensor
    - encoder_padding_mask: a binary Torch.Tensor
    """
    if not isinstance(encoder_output, dict):
        msg = (
            "FairseqEncoderModel.forward(...) must be a dict" + _current_postion_info()
        )
        return False, msg

    if "encoder_out" not in encoder_output:
        msg = (
            "FairseqEncoderModel.forward(...) must contain encoder_out"
            + _current_postion_info()
        )
        return False, msg

    if "encoder_padding_mask" not in encoder_output:
        msg = (
            "FairseqEncoderModel.forward(...) must contain encoder_padding_mask"
            + _current_postion_info()
        )
        return False, msg

    if not isinstance(encoder_output["encoder_out"], torch.Tensor):
        msg = "encoder_out must be a torch.Tensor" + _current_postion_info()
        return False, msg

    if encoder_output["encoder_out"].dtype != torch.float32:
        msg = "encoder_out must have float32 dtype" + _current_postion_info()
        return False, msg

    mask = encoder_output["encoder_padding_mask"]
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            msg = (
                "encoder_padding_mask must be a torch.Tensor" + _current_postion_info()
            )
            return False, msg
        if mask.dtype != torch.uint8 and (
            not hasattr(torch, "bool") or mask.dtype != torch.bool
        ):
            msg = (
                "encoder_padding_mask must have dtype of uint8"
                + _current_postion_info()
            )
            return False, msg

        if mask.dim() != 2:
            msg = (
                "we expect encoder_padding_mask to be a 2-d tensor, in shape (T, B)"
                + _current_postion_info()
            )
            return False, msg

        if batch_size is not None and mask.size(1) != batch_size:
            msg = (
                "we expect encoder_padding_mask to be a 2-d tensor, with size(1)"
                + " being the batch size"
                + _current_postion_info()
            )
            return False, msg
    return True, None


def check_decoder_output(decoder_output):
    """we expect output from a decoder is a tuple with the following constraint:
    - the first element is a torch.Tensor
    - the second element can be anything (reserved for future use)
    """
    if not isinstance(decoder_output, tuple):
        msg = "FariseqDecoder output must be a tuple" + _current_postion_info()
        return False, msg

    if len(decoder_output) != 2:
        msg = "FairseqDecoder output must be 2-elem tuple" + _current_postion_info()
        return False, msg

    if not isinstance(decoder_output[0], torch.Tensor):
        msg = (
            "FariseqDecoder output[0] must be a torch.Tensor" + _current_postion_info()
        )
        return False, msg

    return True, None


# ///////////////////////////////////////////////////////////////////////////
# Base Test class
# ///////////////////////////////////////////////////////////////////////////


class TestBaseFairseqModelBase(unittest.TestCase):
    """
    This class is used to facilitate writing unittest for any class derived from
    `BaseFairseqModel`.
    """

    @classmethod
    def setUpClass(cls):
        if cls is TestBaseFairseqModelBase:
            raise unittest.SkipTest("Skipping test case in base")
        super().setUpClass()

    def setUpModel(self, model):
        self.assertTrue(isinstance(model, BaseFairseqModel))
        self.model = model

    def setupInput(self):
        pass

    def setUp(self):
        self.model = None
        self.forward_input = None
        pass


class TestFairseqEncoderDecoderModelBase(TestBaseFairseqModelBase):
    """
    base code to test FairseqEncoderDecoderModel (formally known as
    `FairseqModel`) must be derived from this base class
    """

    @classmethod
    def setUpClass(cls):
        if cls is TestFairseqEncoderDecoderModelBase:
            raise unittest.SkipTest("Skipping test case in base")
        super().setUpClass()

    def setUpModel(self, model_cls, extra_args_setters=None):
        self.assertTrue(
            issubclass(model_cls, (FairseqEncoderDecoderModel, FairseqModel)),
            msg="This class only tests for FairseqModel subclasses",
        )

        task, parser = get_dummy_task_and_parser()
        model_cls.add_args(parser)

        args = parser.parse_args([])
        if extra_args_setters is not None:
            for args_setter in extra_args_setters:
                args_setter(args)
        model = model_cls.build_model(args, task)
        self.model = model

    def setUpInput(self, input=None):
        self.forward_input = get_dummy_input() if input is None else input

    def setUp(self):
        super().setUp()

    def test_forward(self):
        if self.model and self.forward_input:
            forward_output = self.model.forward(**self.forward_input)
            # for FairseqEncoderDecoderModel, forward returns a tuple of two
            # elements, the first one is a Torch.Tensor
            succ, msg = check_decoder_output(forward_output)
            if not succ:
                self.assertTrue(succ, msg=msg)
            self.forward_output = forward_output

    def test_get_normalized_probs(self):
        if self.model and self.forward_input:
            forward_output = self.model.forward(**self.forward_input)
            logprob = self.model.get_normalized_probs(forward_output, log_probs=True)
            prob = self.model.get_normalized_probs(forward_output, log_probs=False)

            # in order for different models/criterion to play with each other
            # we need to know whether the logprob or prob output is batch_first
            # or not. We assume an additional attribute will be attached to logprob
            # or prob. If you find your code failed here, simply override
            # FairseqModel.get_normalized_probs, see example at
            # https://fburl.com/batch_first_example
            self.assertTrue(hasattr(logprob, "batch_first"))
            self.assertTrue(hasattr(prob, "batch_first"))

            self.assertTrue(torch.is_tensor(logprob))
            self.assertTrue(torch.is_tensor(prob))


class TestFairseqEncoderModelBase(TestBaseFairseqModelBase):
    """
    base class to test FairseqEncoderModel
    """

    @classmethod
    def setUpClass(cls):
        if cls is TestFairseqEncoderModelBase:
            raise unittest.SkipTest("Skipping test case in base")
        super().setUpClass()

    def setUpModel(self, model_cls, extra_args_setters=None):
        self.assertTrue(
            issubclass(model_cls, FairseqEncoderModel),
            msg="This class is only used for testing FairseqEncoderModel",
        )
        task, parser = get_dummy_task_and_parser()
        model_cls.add_args(parser)
        args = parser.parse_args([])
        if extra_args_setters is not None:
            for args_setter in extra_args_setters:
                args_setter(args)

        model = model_cls.build_model(args, task)
        self.model = model

    def setUpInput(self, input=None):
        self.forward_input = get_dummy_input() if input is None else input
        # get_dummy_input() is originally for s2s, here we delete extra dict
        # items, so it can be used for EncoderModel / Encoder as well
        self.forward_input.pop("prev_output_tokens", None)

    def setUp(self):
        super().setUp()

    def test_forward(self):
        if self.forward_input and self.model:
            bsz = self.forward_input["src_tokens"].size(0)
            forward_output = self.model.forward(**self.forward_input)

            # we expect forward_output to be a dict with the following
            # key/value pairs:
            # - encoder_out: a Torch.Tensor
            # - encoder_padding_mask: a binary Torch.Tensor
            succ, msg = check_encoder_output(forward_output, batch_size=bsz)
            if not succ:
                self.assertTrue(succ, msg=msg)
            self.forward_output = forward_output

    def test_get_normalized_probs(self):
        if self.model and self.forward_input:
            forward_output = self.model.forward(**self.forward_input)
            logprob = self.model.get_normalized_probs(forward_output, log_probs=True)
            prob = self.model.get_normalized_probs(forward_output, log_probs=False)

            # in order for different models/criterion to play with each other
            # we need to know whether the logprob or prob output is batch_first
            # or not. We assume an additional attribute will be attached to logprob
            # or prob. If you find your code failed here, simply override
            # FairseqModel.get_normalized_probs, see example at
            # https://fburl.com/batch_first_example
            self.assertTrue(hasattr(logprob, "batch_first"))
            self.assertTrue(hasattr(prob, "batch_first"))

            self.assertTrue(torch.is_tensor(logprob))
            self.assertTrue(torch.is_tensor(prob))


class TestFairseqEncoderBase(unittest.TestCase):
    """
    base class to test FairseqEncoder
    """

    @classmethod
    def setUpClass(cls):
        if cls is TestFairseqEncoderBase:
            raise unittest.SkipTest("Skipping test case in base")
        super().setUpClass()

    def setUpEncoder(self, encoder):
        self.assertTrue(
            isinstance(encoder, FairseqEncoder),
            msg="This class is only used for test FairseqEncoder",
        )
        self.encoder = encoder

    def setUpInput(self, input=None):
        self.forward_input = get_dummy_input() if input is None else input
        # get_dummy_input() is originally for s2s, here we delete extra dict
        # items, so it can be used for EncoderModel / Encoder as well
        self.forward_input.pop("prev_output_tokens", None)

    def setUp(self):
        self.encoder = None
        self.forward_input = None

    def test_forward(self):
        if self.encoder and self.forward_input:
            bsz = self.forward_input["src_tokens"].size(0)

            forward_output = self.encoder.forward(**self.forward_input)
            succ, msg = check_encoder_output(forward_output, batch_size=bsz)
            if not succ:
                self.assertTrue(succ, msg=msg)
            self.forward_output = forward_output


class TestFairseqDecoderBase(unittest.TestCase):
    """
    base class to test FairseqDecoder
    """

    @classmethod
    def setUpClass(cls):
        if cls is TestFairseqDecoderBase:
            raise unittest.SkipTest("Skipping test case in base")
        super().setUpClass()

    def setUpDecoder(self, decoder):
        self.assertTrue(
            isinstance(decoder, FairseqDecoder),
            msg="This class is only used for test FairseqDecoder",
        )
        self.decoder = decoder

    def setUpInput(self, input=None):
        self.forward_input = get_dummy_encoder_output() if input is None else input

    def setUpPrevOutputTokens(self, tokens=None):
        if tokens is None:
            self.encoder_input = get_dummy_input()
            self.prev_output_tokens = self.encoder_input["prev_output_tokens"]
        else:
            self.prev_output_tokens = tokens

    def setUp(self):
        self.decoder = None
        self.forward_input = None
        self.prev_output_tokens = None

    def test_forward(self):
        if (
            self.decoder is not None
            and self.forward_input is not None
            and self.prev_output_tokens is not None
        ):
            forward_output = self.decoder.forward(
                prev_output_tokens=self.prev_output_tokens,
                encoder_out=self.forward_input,
            )
            succ, msg = check_decoder_output(forward_output)
            if not succ:
                self.assertTrue(succ, msg=msg)
            self.forward_input = forward_output


class DummyEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder):
        super().__init__(encoder)

    @classmethod
    def build_model(cls, args, task):
        return cls(DummyEncoder())

    def get_logits(self, net_output):
        # Inverse of sigmoid to use with BinaryCrossEntropyWithLogitsCriterion as
        # F.binary_cross_entropy_with_logits combines sigmoid and CE
        return torch.log(
            torch.div(net_output["encoder_out"], 1 - net_output["encoder_out"])
        )

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        lprobs = super().get_normalized_probs(net_output, log_probs, sample=sample)
        lprobs.batch_first = True
        return lprobs


class DummyEncoder(FairseqEncoder):
    def __init__(self):
        super().__init__(None)

    def forward(self, src_tokens, src_lengths):
        mask, max_len = lengths_to_encoder_padding_mask(src_lengths)
        return {"encoder_out": src_tokens, "encoder_padding_mask": mask}


class CrossEntropyCriterionTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is CrossEntropyCriterionTestBase:
            raise unittest.SkipTest("Skipping base class test case")
        super().setUpClass()

    def setUpArgs(self):
        args = argparse.Namespace()
        args.sentence_avg = False
        args.threshold = 0.1  # to use with BinaryCrossEntropyWithLogitsCriterion
        return args

    def setUp(self):
        args = self.setUpArgs()
        self.model = DummyEncoderModel(encoder=DummyEncoder())
        self.criterion = self.criterion_cls.build_criterion(
            args=args, task=DummyTask(args)
        )

    def get_src_tokens(self, correct_prediction, aggregate):
        """
        correct_prediction: True if the net_output (src_tokens) should
        predict the correct target
        aggregate: True if the criterion expects net_output (src_tokens)
        aggregated across time axis
        """
        predicted_idx = 0 if correct_prediction else 1
        if aggregate:
            src_tokens = torch.zeros((2, 2), dtype=torch.float)
            for b in range(2):
                src_tokens[b][predicted_idx] = 1.0
        else:
            src_tokens = torch.zeros((2, 10, 2), dtype=torch.float)
            for b in range(2):
                for t in range(10):
                    src_tokens[b][t][predicted_idx] = 1.0
        return src_tokens

    def get_target(self, soft_target):
        if soft_target:
            target = torch.zeros((2, 2), dtype=torch.float)
            for b in range(2):
                target[b][0] = 1.0
        else:
            target = torch.zeros((2, 10), dtype=torch.long)
        return target

    def get_test_sample(self, correct, soft_target, aggregate):
        src_tokens = self.get_src_tokens(correct, aggregate)
        target = self.get_target(soft_target)
        L = src_tokens.size(1)
        return {
            "net_input": {"src_tokens": src_tokens, "src_lengths": torch.tensor([L])},
            "target": target,
            "ntokens": src_tokens.size(0) * src_tokens.size(1),
        }
