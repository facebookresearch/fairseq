# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from torch import nn

from fairseq.models import FairseqLanguageModel
from fairseq.utils import buffered_arange


class ElmoTokenEmbedder(nn.Module):
    """
    This is an implementation of the ELMo module which allows learning how to combine hidden states of a language model
    to learn task-specific word representations.
    For more information see the paper here: http://arxiv.org/abs/1802.05365

    This implementation was inspired by the implementation in AllenNLP found here:
    https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
    """

    def __init__(self,
                 language_model: FairseqLanguageModel,
                 eos: int,
                 pad: int,
                 tune_lm: bool = False,
                 weights_dropout: float = 0.,
                 final_dropout: float = 0.,
                 layer_norm: bool = True,
                 affine_layer_norm: bool = False,
                 projection_dim: int = None,
                 apply_softmax: bool = True,
                 combine_tower_states: bool = True,
                 add_final_predictive: bool = True,
                 add_final_context: bool = True,
                 add_bos: bool = False,
                 add_eos: bool = False,
                 remove_bos: bool = False,
                 remove_eos: bool = False,
                 ):
        super().__init__()

        self.language_model = language_model
        self.eos_idx = eos
        self.padding_idx = pad
        self.tune_lm = tune_lm
        self.combine_tower_states = combine_tower_states
        self.add_final_predictive = add_final_predictive
        self.add_final_context = add_final_context
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.remove_bos = remove_bos
        self.remove_eos = remove_eos

        if not tune_lm:
            for param in language_model.parameters():
                param.requires_grad = False
            language_model.eval()

        # pass dummy input through language model to get the number of layers and their dimensions
        dummy_input = next(language_model.parameters()).new_zeros(1, 1).long()
        states = self._lm_states(dummy_input)

        self.num_layers = len(states)
        assert self.num_layers > 0

        self.dim = states[0].size(-1)
        self.embedding_dim = projection_dim or self.dim
        assert all(s.size(-1) == self.dim for s in states[1:])

        self.weights_dropout = nn.Dropout(weights_dropout)
        self.final_dropout = nn.Dropout(final_dropout)
        self.layer_norm = nn.LayerNorm(self.dim, elementwise_affine=affine_layer_norm) if layer_norm else None

        self.weights = nn.Parameter(torch.ones(self.num_layers))
        self.gamma = nn.Parameter(torch.ones(1))

        self.softmax = nn.Softmax(dim=0) if apply_softmax else None

        self.projection = nn.Linear(self.dim, projection_dim,
                                    bias=False) if projection_dim is not None and projection_dim != self.dim else None

    def reset_parameters(self):
        if self.projection:
            nn.init.xavier_normal_(self.projection.weight)
        if self.softmax is None:
            nn.init.constant_(self.weights, 1 / (self.num_layers * 2))

    def reset_parameters(self):
        if self.projection:
            nn.init.xavier_normal_(self.projection.weight)
        if self.softmax is None:
            nn.init.constant_(self.weights, 1 / (self.num_layers * 2))

    def _lm_states(self, input):
        if self.tune_lm:
            _, model_out = self.language_model(input)
        else:
            with torch.no_grad():
                _, model_out = self.language_model(input)

        assert 'inner_states' in model_out

        # TBC -> BTC
        states = [s.transpose(0, 1) for s in model_out['inner_states']]

        has_final_predictive = len(states) % 2 == 0

        if self.add_final_context:
            zeros = states[-1].new_zeros(states[-1].size(0), 1, states[-1].size(2))
            if states[-1].size(1) == 1:
                s1 = s2 = zeros
            else:
                s1 = torch.cat([zeros, states[-1][:, :-1, :]], dim=1)
                s2 = torch.cat([states[-1][:, 1:, :], zeros], dim=1)
            if has_final_predictive:
                states.insert(-1, s1)
                states.insert(-1, s2)
            else:
                states.extend([s1, s2])

        if self.combine_tower_states:
            new_states = [torch.cat([states[0], states[0]], dim=-1)]

            start = 1  # first element is the token embeddings
            end = len(states)
            if has_final_predictive:
                end -= 1

            for i in range(start, end, 2):
                new_states.append(torch.cat([states[i], states[i + 1]], dim=-1))

            if self.add_final_predictive and has_final_predictive:
                new_states.append(torch.cat([states[-1], states[-1]], dim=-1))

            states = new_states
        elif not self.add_final_predictive and has_final_predictive:
                states = states[:-1]

        return states

    def _with_sentence_boundaries(
            self,
            input: torch.Tensor,
    ):
        zero_block = input.new(0, 0)
        bos_block = input.new_full((input.size(0), 1), self.eos_idx) if self.add_bos else zero_block
        pad_block = input.new_full((input.size(0), 1), self.padding_idx) if self.add_eos else zero_block

        # add eos in the beginning and pad to the end of the sentence
        input = torch.cat([bos_block, input, pad_block], dim=1)

        if self.add_eos:
            num_pads = input.eq(self.padding_idx).long().sum(dim=1, keepdim=True)
            max_len = input.size(1)

            # index of the first pad
            first_pads = buffered_arange(max_len).view(1, -1).expand(input.size(0), -1).eq(max_len - num_pads)
            input[first_pads] = self.eos_idx

        return input

    def _without_sentence_boundaries(
            self,
            input: torch.Tensor,
    ):
        if self.remove_bos:
            # remove first token (beginning eos)
            input = input[:, 1:]
        if self.remove_eos:
            # turn end eos into pads
            input[input.eq(self.eos_idx)] = self.padding_idx
            # remove last pad and return
            input = input[:, :-1]
        return input

    def forward(
            self,
            input: torch.Tensor,
    ):
        input = self._with_sentence_boundaries(input)

        states = self._lm_states(input)

        if self.layer_norm is not None:
            states = [self.layer_norm(s) for s in states]

        if self.softmax is not None:
            w = self.softmax(self.weights)
        else:
            w = self.weights

        w = self.weights_dropout(w)

        x = states[0].new_zeros(input.size() + (self.dim,))
        for i in range(len(states)):
            x += states[i] * w[i]

        x = self._without_sentence_boundaries(x)

        if self.projection is not None:
            x = self.projection(x)

        x = self.gamma * x

        x = self.final_dropout(x)

        return x
