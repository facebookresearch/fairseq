# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch.autograd import Variable

from fairseq import data, dictionary, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqModel,
)


def dummy_dictionary(vocab_size, prefix='token_'):
    d = dictionary.Dictionary()
    for i in range(vocab_size):
        token = prefix + str(i)
        d.add_symbol(token)
    d.finalize()
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
        collate_fn=(
            lambda samples: data.LanguagePairDataset.collate(
                samples,
                padding_idx,
                eos_idx,
            )
        ),
    )
    return iter(dataloader)


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, src_dict, dst_dict):
        encoder = TestEncoder(args, src_dict)
        decoder = TestIncrementalDecoder(args, dst_dict)
        return cls(encoder, decoder)


class TestEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, src_tokens, src_lengths):
        return src_tokens


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
        attn = torch.rand(bbsz, src_len, tgt_len)

        return Variable(probs), Variable(attn)

    def get_normalized_probs(self, net_output, log_probs):
        # the decoder returns probabilities directly
        probs = net_output[0]
        if log_probs:
            return probs.log()
        else:
            return probs

    def max_positions(self):
        return self.args.max_decoder_positions
