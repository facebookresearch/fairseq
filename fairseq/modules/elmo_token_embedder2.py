# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch

import torch.nn.functional as F

from torch import nn

from fairseq.models import FairseqLanguageModel
from fairseq.utils import buffered_arange


class LearningToNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.dims = dims
        self.gamma = nn.Parameter(torch.ones(1))
        self.mean_weights = nn.Parameter(torch.zeros(dims + 1))
        self.var_weights = nn.Parameter(torch.zeros(dims + 1))
        self.sm = nn.Softmax(dim=-1)
        self.eps = 1e-12

    def forward(self, input: torch.Tensor, mask: torch.Tensor):

        if mask is None:
            mask = input.new_ones(input.shape)
        else:
            while mask.dim() < input.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(input).type_as(input)

        mean_weights = self.sm(self.mean_weights)
        var_weights = self.sm(self.var_weights)

        masked_input = (input * mask).contiguous()

        mean = masked_input.new_zeros(masked_input.shape[:-1] + (1,))
        var = masked_input.new_full(masked_input.shape[:-1] + (1,), var_weights[0].item())

        for i in range(1, self.dims + 1):
            shape = masked_input.shape[:-i] + (-1,)
            vw = masked_input.view(shape)
            mask_vw = mask.view(shape)
            num = mask_vw.sum(dim=-1, keepdim=True) + self.eps
            curr_mean = vw.sum(dim=-1, keepdim=True) / num
            diff = (vw - curr_mean)
            diff = diff * mask_vw
            curr_var = (diff.pow(2).sum(dim=-1, keepdim=True) / (num - 1))

            final_shape = masked_input.shape[:-i] + (1,) * i

            mean += mean_weights[i] * curr_mean.view(final_shape)
            var += var_weights[i] * curr_var.view(final_shape)

        return self.gamma * (input - mean) / (var + self.eps).sqrt()

        # return (input - mean) / (var + self.eps).sqrt()


class ElmoTokenEmbedder2(nn.Module):
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
                 channelwise_weights=False,
                 scaled_sigmoid=False,
                 individual_norms=False,
                 channelwise_norm=False,
                 init_gamma=1.0,
                 ltn=False,
                 ltn_dims=None,
                 train_gamma=True,
                 add_intermediate_context=False,
                 ):
        super().__init__()

        self.language_model = language_model
        #self.language_model.decoder.embed_tokens.disable_convolutional_grads()
        #for n, p in self.language_model.named_parameters():
        #    if 'layers.0' in n or 'layers.1' in n or 'layers.2' in n:
        #        p.requires_grad = False
        self.eos_idx = eos
        self.padding_idx = pad
        self.tune_lm = tune_lm
        self.combine_tower_states = combine_tower_states
        self.add_final_predictive = add_final_predictive
        self.add_final_context = add_final_context
        self.add_intermediate_context = add_intermediate_context
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.remove_bos = remove_bos
        self.remove_eos = remove_eos
        self.individual_norms = individual_norms
        self.channelwise_norm = channelwise_norm
        self.batch_norm_fused = False

        self.ltn = None

        if not tune_lm:
            for param in language_model.parameters():
                param.requires_grad = False
            language_model.eval()

        # pass dummy input through language model to get the number of layers and their dimensions
        dummy_input = next(language_model.parameters()).new_zeros(1, 1).long()
        
        states, num_raw_states = self._lm_states(dummy_input)
        
        self.num_layers = len(states)
        assert self.num_layers > 0

        self.dim = states[0].size(-1)
        self.embedding_dim = projection_dim or self.dim
        assert all(s.size(-1) == self.dim for s in states[1:])

        print(f'elmo {self.num_layers} x {self.dim}')

        self.weights_dropout = nn.Dropout(weights_dropout)
        self.final_dropout = nn.Dropout(final_dropout)

        self.layer_norm = None
        if layer_norm:
            sz = self.num_layers if channelwise_norm and not ltn else self.dim
            if individual_norms:
                assert affine_layer_norm
                self.layer_norm = nn.ModuleList(
                    nn.LayerNorm(sz, elementwise_affine=affine_layer_norm) for _ in range(self.num_layers)
                )
            else:
                self.layer_norm = nn.LayerNorm(sz, elementwise_affine=affine_layer_norm)

        self.channelwise_weights = channelwise_weights
        self.instance_norm = None
        self.batch_norm = None

        self.weights = None
        self.softmax = None

        if channelwise_weights:
            self.weights = nn.Parameter(torch.ones(self.dim, self.num_layers))
        else:
            self.weights = nn.Parameter(torch.Tensor(self.num_layers).fill_(1.0)) if not self.tune_lm else None
        self.softmax = nn.Softmax(dim=-1) if apply_softmax else None

        self.sigmoid_weights = nn.Parameter(torch.zeros(self.num_layers, self.dim)) if scaled_sigmoid else None

        self.gamma = nn.Parameter(torch.full((1,), init_gamma), requires_grad=train_gamma) if not self.tune_lm else None
        self.projection = nn.Linear(self.dim, self.embedding_dim,
                                    bias=False) if self.embedding_dim != self.dim else None
        if ltn:
            if self.individual_norms:
                ltn_dims = ltn_dims or 3
                assert ltn_dims <= 3
                self.ltn = nn.ModuleList(
                    LearningToNorm(dims=ltn_dims) for _ in range(self.num_layers)
                )
            else:
                ltn_dims = ltn_dims or 4
                assert ltn_dims <= 4
                self.ltn = LearningToNorm(dims=ltn_dims)

    def reset_parameters(self):
        if self.projection:
            nn.init.xavier_uniform_(self.projection.weight)
        if self.softmax is None:
            nn.init.constant_(self.weights, 1 / (self.num_layers * 2))
        if self.scaled_sigmoid:
            nn.init.init.constant_(self.sigmoid_weights, 0)
        if self.megaproj:
            for m in self.megaproj:
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def _lm_states(self, input):
        if self.tune_lm:
            model_out, _ = self.language_model(input, src_lengths=None)
            model_out = [model_out]
            return model_out, 1
        else:
            with torch.no_grad():
                if self.language_model.training:
                    self.language_model.eval()
                _, model_out = self.language_model(input, src_lengths=None)

        assert 'inner_states' in model_out

        # TBC -> BTC
        states = [s.transpose(0, 1) for s in model_out['inner_states']]

        has_final_predictive = len(states) % 2 == 0

        new_states = []
        if self.add_intermediate_context:
            zeros = states[-1].new_zeros(states[-1].size(0), 1, states[-1].size(2))

            start = 1  # first element is the token embeddings
            end = len(states)
            if has_final_predictive:
                end -= 1

            for i in range(start, end, 2):
                if states[i].size(1) == 1:
                    s1 = s2 = zeros
                else:
                    # fwd is before bwd
                    s1 = torch.cat([zeros, states[i][:, :-1, :]], dim=1)
                    s2 = torch.cat([states[i + 1][:, 1:, :], zeros], dim=1)
                new_states.extend([s1, s2])

        if self.add_final_context and has_final_predictive:
            zeros = states[-1].new_zeros(states[-1].size(0), 1, states[-1].size(2))
            if states[-1].size(1) == 1:
                s1 = s2 = zeros
            else:
                s1 = torch.cat([zeros, states[-1][:, :-1, :]], dim=1)
                s2 = torch.cat([states[-1][:, 1:, :], zeros], dim=1)
            new_states.extend([s1, s2])

        if has_final_predictive:
            states[-1:-1] = new_states
        else:
            states.extend(new_states)

        if self.combine_tower_states:

            def combine(s1, s2):
                return torch.cat([s1, s2], dim=-1)

            new_states = [combine(states[0], states[0])]

            start = 1  # first element is the token embeddings
            end = len(states)
            if has_final_predictive:
                end -= 1

            for i in range(start, end, 2):
                new_states.append(combine(states[i], states[i + 1]))

            if self.add_final_predictive and has_final_predictive:
                new_states.append(combine(states[-1], states[-1]))

            states = new_states
        elif not self.add_final_predictive and has_final_predictive:
            states = states[:-1]

        return states, len(model_out['inner_states'])

    def _with_sentence_boundaries(
            self,
            input: torch.Tensor,
    ):
        if not self.add_bos and not self.add_eos:
            return input

        zero_block = input.new(input.size(0), 0)
        bos_block = input.new_full((input.size(0), 1), self.eos_idx) if self.add_bos else zero_block
        pad_block = input.new_full((input.size(0), 1), self.padding_idx) if self.add_eos else zero_block

        # add eos in the beginning and pad to the end of the sentence
        input = torch.cat([bos_block, input, pad_block], dim=1)

        if self.add_eos:
            num_pads = input.eq(self.padding_idx).long().sum(dim=1, keepdim=True)
            max_len = input.size(1)

            # index of the first pad
            first_pads = buffered_arange(max_len).type_as(input).view(1, -1).expand(input.size(0), -1).eq(
                max_len - num_pads)
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

        if self.tune_lm:
            states, _  = self.language_model(input, src_lengths=None)
            
            states = [states]
        else:
            states, _ = self._lm_states(input)
        if self.layer_norm is not None:
            if self.ltn is None and self.channelwise_norm:
                states = torch.stack(states, dim=1).transpose(1, 3)
                states = self.layer_norm(states)
                states = states.transpose(1, 3)
                states = [x.squeeze(1) for x in torch.split(states, 1, dim=1)]
            elif self.individual_norms:
                states = [self.layer_norm[i](states[i]) for i in range(len(states))]
            else:
                states = [self.layer_norm(s) for s in states]

        if self.ltn is not None:
            mask = input.ne(self.padding_idx)

            if self.channelwise_norm:
                mask = mask.unsqueeze(1)
                states = torch.stack(states, dim=1)
                if self.individual_norms:
                    states = states.transpose(2, 3)
                    for i in range(len(self.ltn)):
                        states[:, i] = self.ltn[i](states[:, i], mask)
                    states = states.transpose(2, 3)
                else:
                    states = states.transpose(1, 3)
                    states = self.ltn(states, mask)
                    states = states.transpose(1, 3)
                states = [x.squeeze(1) for x in torch.split(states, 1, dim=1)]
            else:
                if self.individual_norms:
                    states = [self.ltn[i](states[i], mask) for i in range(len(states))]
                else:
                    states = torch.stack(states, dim=1)
                    mask = mask.unsqueeze(1)
                    states = self.ltn(states, mask)
                    states = [x.squeeze(1) for x in torch.split(states, 1, dim=1)]

        if self.softmax is not None and self.weights is not None:
            w = self.softmax(self.weights)
        else:
            w = self.weights

        if self.channelwise_weights and w is not None:
            w = w.t()

        if w is not None:
            w = self.weights_dropout(w)

        x = states[0].new_zeros(input.size() + (self.dim,))
        for i in range(len(states)):
            s = states[i]
            if self.sigmoid_weights is not None:
                sw = F.sigmoid(self.sigmoid_weights[i]) * 2
                s = s * sw
            if w is not None:
                x += s * w[i]
            else:
                x += s
        if self.tune_lm:
            assert (w is None)                

        x = self._without_sentence_boundaries(x)

        if self.projection is not None:
            x = self.projection(x)

        if self.gamma:
            x = self.gamma * x

        x = self.final_dropout(x)

        return x
