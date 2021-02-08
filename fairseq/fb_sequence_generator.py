# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
from fairseq.sequence_generator import EnsembleModel
from fairseq.sequence_generator import SequenceGenerator
from torch import Tensor


class FBEnsembleModelWithFork(EnsembleModel):
    """A wrapper around the ~fairseq.sequence_generator.EnsembleModel to support
    fork/join.
    """

    def __init__(self, models):
        super().__init__(models)

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None

        futures = [
            torch.jit._fork(model.encoder.forward_torchscript, net_input)
            for model in self.models
        ]
        return [torch.jit._wait(fut) for fut in futures]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        futures = [
            torch.jit._fork(
                model.decoder,
                tokens,
                encoder_outs[i] if self.has_encoder() else None,
                incremental_states[i],
            )
            for i, model in enumerate(self.models)
        ]
        log_probs = []
        avg_attn: Optional[Tensor] = None
        for i, model in enumerate(self.models):
            decoder_out = torch.jit._wait(futures[i])
            attn: Optional[Tensor] = None
            # Future type doesn't support len().
            decoder_len = 0
            for _ in decoder_out:
                decoder_len += 1
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :] / temperature,
                None if decoder_len <= 1 else decoder_out[1],
            )
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]

            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn = avg_attn + attn
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )
        if avg_attn is not None:
            avg_attn = avg_attn / self.models_size
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )


class FBSequenceGenerator(SequenceGenerator):
    pass
