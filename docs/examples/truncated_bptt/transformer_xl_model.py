# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from omegaconf import II


logger = logging.getLogger(__name__)


@dataclass
class TransformerXLConfig(FairseqDataclass):
    # defaults come from the original Transformer-XL code
    cutoffs: List[int] = field(default_factory=lambda: [20000, 40000, 200000])
    d_model: int = 500
    n_head: int = 10
    d_head: int = 50
    d_inner: int = 1000
    div_val: int = 1
    n_layer: int = 12
    mem_len: int = 0
    clamp_len: int = -1
    same_length: bool = False
    dropout: float = 0.0
    dropatt: float = 0.0
    checkpoint_activations: bool = False
    offload_activations: bool = False
    max_target_positions: int = II("task.max_target_positions")


@register_model("transformer_xl", dataclass=TransformerXLConfig)
class TransformerXLLanguageModel(FairseqLanguageModel):
    @classmethod
    def build_model(cls, cfg: TransformerXLConfig, task):
        return cls(TransformerXLDecoder(cfg, task))


class TransformerXLDecoder(FairseqIncrementalDecoder):
    def __init__(self, cfg, task):
        try:
            from transformers.models.transfo_xl import (
                TransfoXLConfig,
                TransfoXLLMHeadModel,
            )
        except ImportError:
            from transformers.configuration_transfo_xl import TransfoXLConfig
            from transformers.modeling_transfo_xl import TransfoXLLMHeadModel

        super().__init__(task.target_dictionary)
        self.cfg = cfg

        # remove any cutoffs larger than the vocab size
        cutoffs = [
            cutoff for cutoff in cfg.cutoffs if cutoff < len(task.target_dictionary)
        ]

        config = TransfoXLConfig(
            vocab_size=len(task.target_dictionary),
            cutoffs=cutoffs,
            d_model=cfg.d_model,
            d_embed=cfg.d_model,
            n_head=cfg.n_head,
            d_head=cfg.d_head,
            d_inner=cfg.d_inner,
            div_val=cfg.div_val,
            n_layer=cfg.n_layer,
            mem_len=cfg.mem_len,
            clamp_len=cfg.clamp_len,
            same_length=cfg.same_length,
            dropout=cfg.dropout,
            dropatt=cfg.dropatt,
        )
        logger.info(config)
        self.model = TransfoXLLMHeadModel(config)

        if cfg.checkpoint_activations or cfg.offload_activations:
            for i in range(len(self.model.transformer.layers)):
                self.model.transformer.layers[i] = checkpoint_wrapper(
                    self.model.transformer.layers[i],
                    offload_to_cpu=cfg.offload_activations,
                )
                # TODO: may save mem to wrap(layer.pos_ff.CoreNet[3])

        self._mems = None

    def forward(
        self,
        src_tokens,
        src_lengths=None,  # unused
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
    ):
        if incremental_state is not None:  # used during inference
            mems = self.get_incremental_state(incremental_state, "mems")
            src_tokens = src_tokens[:, -1:]  # only keep the most recent token
        else:
            mems = self._mems

        output = self.model(
            input_ids=src_tokens,
            mems=mems,
            return_dict=False,
        )

        if len(output) >= 2:
            if incremental_state is not None:
                self.set_incremental_state(incremental_state, "mems", output[1])
            else:
                self._mems = output[1]

        return (output[0],)

    def max_positions(self):
        return self.cfg.max_target_positions

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        new_order: torch.Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        mems = self.get_incremental_state(incremental_state, "mems")
        if mems is not None:
            new_mems = [mems_i.index_select(1, new_order) for mems_i in mems]
            self.set_incremental_state(incremental_state, "mems", new_mems)
