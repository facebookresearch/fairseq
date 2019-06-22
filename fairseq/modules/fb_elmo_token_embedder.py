# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from typing import Dict, List

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

    def __init__(
        self,
        language_model: FairseqLanguageModel,
        eos: int,
        pad: int,
        tune_lm: bool = False,
        lm_frozen_layers: int = 0,
        lm_tune_embedding: bool = False,
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
        char_inputs: bool = False,
        max_char_len: int = 50,
        use_boundary_tokens: bool = False,
    ):

        super().__init__()

        self.onnx_trace = False
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
        self.char_inputs = char_inputs
        # use_boundary_tokens will only use the bos/eos of the ELMO last layer,
        # will override some other options in _lm_states and forward,
        # for the purpose of fine-tuning the language model
        self.use_boundary_tokens = use_boundary_tokens

        if self.use_boundary_tokens:
            # make sure the bos and eos are not remove in fine tuning case
            assert (not self.remove_bos)
            assert (not self.remove_eos)

        self.num_layers = len(language_model.decoder.forward_layers)
        if self.add_final_context:
            self.num_layers += 1
        if not self.combine_tower_states:
            self.num_layers *= 2
        # +1 for token embedding layer
        self.num_layers += 1
        if language_model.decoder.self_target and self.add_final_predictive:
            self.num_layers += 1

        self.dim = language_model.decoder.embed_dim
        if not self.use_boundary_tokens and self.combine_tower_states:
            self.dim *= 2
        self.embedding_dim = projection_dim or self.dim

        self.weights_dropout = nn.Dropout(weights_dropout)
        self.final_dropout = nn.Dropout(final_dropout)
        self.layer_norm = nn.LayerNorm(self.dim, elementwise_affine=affine_layer_norm) if layer_norm else None

        if self.use_boundary_tokens:
            self.weights = None
            self.gamma = None
        else:
            self.weights = nn.Parameter(torch.ones(self.num_layers))
            self.gamma = nn.Parameter(torch.ones(1))

        self.apply_softmax = apply_softmax

        self.projection = nn.Linear(self.dim, projection_dim,
                                    bias=False) if projection_dim is not None and projection_dim != self.dim else None

        trainable_params, non_trainable_params = self._get_params_by_trainability(
            lm_frozen_layers, lm_tune_embedding
        )

        self.trainable_params_by_layer: List[Dict[str, nn.Parameter]] = trainable_params
        for p in non_trainable_params:
            p.requires_grad = False
        if not tune_lm:
            language_model.eval()

    def _get_params_by_trainability(self, lm_frozen_layers, lm_tune_embedding):
        non_lm_params = self._non_lm_parameters()

        if not self.tune_lm:
            # Only non-lm parameters are trainable
            return [non_lm_params], self.language_model.parameters()

        if not hasattr(self.language_model, "get_layers_by_depth_for_fine_tuning"):
            assert lm_frozen_layers == 0
            # All params are trainable
            return [dict(self.named_parameters())], []

        lm_params_by_layer = self._lm_parameters_by_layer()
        assert len(lm_params_by_layer) >= lm_frozen_layers + 1  # +1 for embedding

        trainable_lm_params = []
        non_trainable_lm_params = []

        if lm_tune_embedding:
            trainable_lm_params.append(lm_params_by_layer[0])
        else:
            non_trainable_lm_params.append(lm_params_by_layer[0])

        trainable_lm_params.extend(lm_params_by_layer[lm_frozen_layers + 1:])
        non_trainable_lm_params.extend(lm_params_by_layer[1: lm_frozen_layers + 1])

        trainable_params = trainable_lm_params + [non_lm_params]
        non_trainable_params = [
            p for param_dict in non_trainable_lm_params for p in param_dict.values()
        ]
        return trainable_params, non_trainable_params

    def _non_lm_parameters(self):
        non_lm_parameters = dict(self.named_parameters())
        for name, _ in self.language_model.named_parameters():
            del non_lm_parameters["language_model.%s" % name]
        return non_lm_parameters

    def _lm_parameters_by_layer(self):
        lm_layers = self.language_model.get_layers_by_depth_for_fine_tuning()
        return [
            {
                "language_model.%s.%s" % (module_name, param_name): param
                for module_name, module in lm_layer.items()
                for param_name, param in module.named_parameters()
            }
            for lm_layer in lm_layers
        ]

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.projection:
            nn.init.xavier_normal_(self.projection.weight)
        if self.softmax is None:
            nn.init.constant_(self.weights, 1 / (self.num_layers * 2))

    def _lm_states(self, input: torch.Tensor, eos_idx_mask=None):
        """apply the language model on the input and get internal states
        Args:
            input: the sentence tensor
            eos_idx_mask: the mask for the index of eos for each sentence

        Returns:
            return a list of states from the language model,
            if use_boundary_tokens, only return the last layer
            if combine_tower_states, will combine forward and backward
        """
        if self.tune_lm:
            x, model_out = self.language_model(input, src_lengths=None)
        else:
            with torch.no_grad():
                x, model_out = self.language_model(input, src_lengths=None)

        if self.use_boundary_tokens:
            bos_state = x[:, 0, :]
            if eos_idx_mask is None:
                return [bos_state.unsqueeze(1)]
            eos_state = x[eos_idx_mask]  # batch_size * embeding_size
            return [torch.cat((bos_state.unsqueeze(1), eos_state.unsqueeze(1)), dim=1)]

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
            input: torch.Tensor):
        """
        Args:
            input: the sentence Tensor
            it's bs * seq_len * num_chars in case of char input and bs*seq_len in case of token input

        Returns:
            tuple,
                1) processed input,
                2) tensor mask for the eos position of each sentence,
                    None if did not add eos
        """
        if not self.add_bos and not self.add_eos:
            return input, None

        zero_block = input.new(0, 0)
        block_size = (input.size(0), 1, input.size(2)) if self.char_inputs else (input.size(0), 1)
        bos_block = torch.full(block_size, self.eos_idx).type_as(input) if self.add_bos else zero_block
        pad_block = torch.full(block_size, self.padding_idx).type_as(input) if self.add_eos else zero_block

        # add eos in the beginning and pad to the end of the sentence
        input = torch.cat([bos_block, input, pad_block], dim=1)

        first_pads = None  # if not add_eos, then first_pads is not valid, set to None
        if self.add_eos:
            index_block = input[:, :, 0] if self.char_inputs else input
            padding_mask = index_block.eq(self.padding_idx)
            num_pads = padding_mask.long().sum(dim=1, keepdim=True)
            max_len = input.size(1)

            # index of the first pad
            if self.onnx_trace:
                first_pads = torch._dim_arange(input, 1).type_as(input).view(1, -1).\
                    repeat(input.size(0), 1).eq(max_len - num_pads)
                eos_indices = first_pads
                if self.char_inputs:
                    eos_indices = eos_indices.unsqueeze(2).repeat(1, 1, input.size(-1))
                input = torch.where(eos_indices, torch.Tensor([self.eos_idx]).type_as(input), input)
            else:
                first_pads = buffered_arange(max_len).type_as(input).view(1, -1).\
                    expand(input.size(0), -1).eq(max_len - num_pads)
                eos_indices = first_pads
                if self.char_inputs:
                    eos_indices = eos_indices.unsqueeze(2).expand_as(input)
                input[eos_indices] = self.eos_idx

        return input, first_pads

    def _without_sentence_boundaries(
            self,
            input: torch.Tensor,
    ):
        if self.remove_bos:
            # remove first token (beginning eos)
            input = input[:, 1:]
        if self.remove_eos:
            # just remove last one to match size since downstream task
            # needs to deal with padding value
            input = input[:, :-1]
        return input

    def forward(
            self,
            input: torch.Tensor,
    ):
        input, eos_idx_mask = self._with_sentence_boundaries(input)

        states = self._lm_states(input, eos_idx_mask)

        if self.use_boundary_tokens:
            return states[0]  # only have one element and return it

        if self.layer_norm is not None:
            states = [self.layer_norm(s) for s in states]

        if self.apply_softmax:
            w = torch.nn.functional.softmax(
                self.weights, dim=0, dtype=torch.float32).type_as(self.weights)
        else:
            w = self.weights

        w = self.weights_dropout(w)

        x = states[0].new_zeros(input.size()[:2] + (self.dim,))
        for i in range(len(states)):
            x += states[i] * w[i]

        x = self._without_sentence_boundaries(x)

        if self.projection is not None:
            x = self.projection(x)

        x = self.gamma * x

        x = self.final_dropout(x)

        return x
