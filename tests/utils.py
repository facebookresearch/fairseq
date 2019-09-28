# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse
import torch

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.data.language_pair_dataset import collate
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqModel,
)
from fairseq.tasks import FairseqTask


def dummy_dictionary(vocab_size, prefix='token_'):
    d = Dictionary()
    for i in range(vocab_size):
        token = prefix + str(i)
        d.add_symbol(token)
    d.finalize(padding_factor=1)  # don't add extra padding symbols
    return d


def dummy_dataloader(
    samples,
    padding_idx=1,
    eos_idx=2,
    batch_size=None,
):
    if batch_size is None:
        batch_size = len(samples)

    # add any missing data to samples
    for i, sample in enumerate(samples):
        if 'id' not in sample:
            sample['id'] = i

    # create dataloader
    dataset = TestDataset(samples)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=(lambda samples: collate(samples, padding_idx, eos_idx)),
    )
    return iter(dataloader)


def sequence_generator_setup():
    # construct dummy dictionary
    d = dummy_dictionary(vocab_size=2)

    eos = d.eos()
    w1 = 4
    w2 = 5

    # construct source data
    src_tokens = torch.LongTensor([[w1, w2, eos], [w1, w2, eos]])
    src_lengths = torch.LongTensor([2, 2])

    args = argparse.Namespace()
    unk = 0.
    args.beam_probs = [
        # step 0:
        torch.FloatTensor([
            # eos      w1   w2
            # sentence 1:
            [0.0, unk, 0.9, 0.1],  # beam 1
            [0.0, unk, 0.9, 0.1],  # beam 2
            # sentence 2:
            [0.0, unk, 0.7, 0.3],
            [0.0, unk, 0.7, 0.3],
        ]),
        # step 1:
        torch.FloatTensor([
            # eos      w1   w2       prefix
            # sentence 1:
            [1.0, unk, 0.0, 0.0],  # w1: 0.9  (emit: w1 <eos>: 0.9*1.0)
            [0.0, unk, 0.9, 0.1],  # w2: 0.1
            # sentence 2:
            [0.25, unk, 0.35, 0.4],  # w1: 0.7  (don't emit: w1 <eos>: 0.7*0.25)
            [0.00, unk, 0.10, 0.9],  # w2: 0.3
        ]),
        # step 2:
        torch.FloatTensor([
            # eos      w1   w2       prefix
            # sentence 1:
            [0.0, unk, 0.1, 0.9],  # w2 w1: 0.1*0.9
            [0.6, unk, 0.2, 0.2],  # w2 w2: 0.1*0.1  (emit: w2 w2 <eos>: 0.1*0.1*0.6)
            # sentence 2:
            [0.60, unk, 0.4, 0.00],  # w1 w2: 0.7*0.4  (emit: w1 w2 <eos>: 0.7*0.4*0.6)
            [0.01, unk, 0.0, 0.99],  # w2 w2: 0.3*0.9
        ]),
        # step 3:
        torch.FloatTensor([
            # eos      w1   w2       prefix
            # sentence 1:
            [1.0, unk, 0.0, 0.0],  # w2 w1 w2: 0.1*0.9*0.9  (emit: w2 w1 w2 <eos>: 0.1*0.9*0.9*1.0)
            [1.0, unk, 0.0, 0.0],  # w2 w1 w1: 0.1*0.9*0.1  (emit: w2 w1 w1 <eos>: 0.1*0.9*0.1*1.0)
            # sentence 2:
            [0.1, unk, 0.5, 0.4],  # w2 w2 w2: 0.3*0.9*0.99  (emit: w2 w2 w2 <eos>: 0.3*0.9*0.99*0.1)
            [1.0, unk, 0.0, 0.0],  # w1 w2 w1: 0.7*0.4*0.4  (emit: w1 w2 w1 <eos>: 0.7*0.4*0.4*1.0)
        ]),
    ]

    task = TestTranslationTask.setup_task(args, d, d)
    model = task.build_model(args)
    tgt_dict = task.target_dictionary

    return tgt_dict, w1, w2, src_tokens, src_lengths, model


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestTranslationTask(FairseqTask):

    def __init__(self, args, src_dict, tgt_dict, model):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.model = model

    @classmethod
    def setup_task(cls, args, src_dict=None, tgt_dict=None, model=None):
        return cls(args, src_dict, tgt_dict, model)

    def build_model(self, args):
        return TestModel.build_model(args, self)

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict


class TestModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        encoder = TestEncoder(args, task.source_dictionary)
        decoder = TestIncrementalDecoder(args, task.target_dictionary)
        return cls(encoder, decoder)


class TestEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, src_tokens, src_lengths):
        return src_tokens

    def reorder_encoder_out(self, encoder_out, new_order):
        return encoder_out.index_select(0, new_order)


class TestIncrementalDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        assert hasattr(args, 'beam_probs') or hasattr(args, 'probs')
        args.max_decoder_positions = getattr(args, 'max_decoder_positions', 100)
        self.args = args

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bbsz = prev_output_tokens.size(0)
        vocab = len(self.dictionary)
        src_len = encoder_out.size(1)
        tgt_len = prev_output_tokens.size(1)

        # determine number of steps
        if incremental_state is not None:
            # cache step number
            step = utils.get_incremental_state(self, incremental_state, 'step')
            if step is None:
                step = 0
            utils.set_incremental_state(self, incremental_state, 'step', step + 1)
            steps = [step]
        else:
            steps = list(range(tgt_len))

        # define output in terms of raw probs
        if hasattr(self.args, 'probs'):
            assert self.args.probs.dim() == 3, \
                'expected probs to have size bsz*steps*vocab'
            probs = self.args.probs.index_select(1, torch.LongTensor(steps))
        else:
            probs = torch.FloatTensor(bbsz, len(steps), vocab).zero_()
            for i, step in enumerate(steps):
                # args.beam_probs gives the probability for every vocab element,
                # starting with eos, then unknown, and then the rest of the vocab
                if step < len(self.args.beam_probs):
                    probs[:, i, self.dictionary.eos():] = self.args.beam_probs[step]
                else:
                    probs[:, i, self.dictionary.eos()] = 1.0

        # random attention
        attn = torch.rand(bbsz, tgt_len, src_len)

        return probs, attn

    def get_normalized_probs(self, net_output, log_probs, _):
        # the decoder returns probabilities directly
        probs = net_output[0]
        if log_probs:
            return probs.log()
        else:
            return probs

    def max_positions(self):
        return self.args.max_decoder_positions
