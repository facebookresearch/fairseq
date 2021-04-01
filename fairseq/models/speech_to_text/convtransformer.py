#!/usr/bin/env python3

import logging
import math
from omegaconf import II
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq import checkpoint_utils, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerEncoderLayer
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ConvTransformerModelConfig(FairseqDataclass):

    conv_out_channels: int = field(
        default=512,
        metadata={"help": "the number of output channels of conv layer"}
    )
    encoder_embed_dim: int = field(
        default=512,
        metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_layers: int = field(
        default=6,
        metadata={"help": "num encoder layers"}
    )
    encoder_attention_heads: int = field(
        default=8,
        metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each encoder block"}
    )
    decoder_embed_dim: int = field(
        default=512,
        metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6,
        metadata={"help": "num decoder layers"}
    )
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN."}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu",
        metadata={"help": "activation function to use"}
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of adaptive softmax cutoff "
                          "points. Must be used with adaptive_loss criterion"}
    )
    adaptive_softmax_dropout: float = field(
        default=0.0,
        metadata={"help": "sets adaptive softmax dropout for "
                          "the tail projections"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={"help": "if set, disables positional embeddings "
                          "(outside self attention)"}
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    decoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_output_dim: int = field(
        default=512,
        metadata={"help": "decoder output dimension (extra linear layer "
                          "if different from decoder embed dim)"}
    )
    decoder_input_dim: int = field(
        default=512,
        metadata={"help": "decoder input dimension"}
    )
    no_scale_embedding: bool = field(
        default=False,
        metadata={"help": "if True, dont scale embeddings"}
    )
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"}
    )
    layernorm_embedding: bool = field(
        default=False,
        metadata={"help": "if True, add layernorm to embedding"}
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={"help": "if set, ties the weights of adaptive softmax "
                          "and adaptive input"},
    )
    load_pretrained_encoder_from: Optional[str] = field(
        default=None,
        metadata={"help": "model to take encoder weights from "
                          "(for initialization)"}
    )
    load_pretrained_decoder_from: Optional[str] = field(
        default=None,
        metadata={"help": "model to take decoder weights from "
                          "(for initialization)"}
    )

    # Inherit from other configs
    max_source_positions: int = II("task.max_source_positions")
    max_target_positions: int = II("task.max_target_positions")

    # Populated in build_model()
    input_feat_per_channel: Optional[int] = field(
        default=None,
        metadata={"help": "dimension of input features (per audio channel)"}
    )


@register_model("convtransformer", dataclass=ConvTransformerModelConfig)
class ConvTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer-based Speech translation model from ESPNet-ST
    https://arxiv.org/abs/2004.10234
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg):
        encoder = ConvTransformerEncoder(cfg)
        if getattr(cfg, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=cfg.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, cfg, task, embed_tokens):
        decoder = TransformerDecoderNoExtra(
            cfg, task.target_dictionary, embed_tokens
        )
        if getattr(cfg, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=cfg.load_pretrained_decoder_from
            )
        return decoder

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        cfg.input_feat_per_channel = task.data_cfg.input_feat_per_channel

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, cfg.decoder_embed_dim
        )
        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    @staticmethod
    @torch.jit.unused
    def set_batch_first(lprobs):
        lprobs.batch_first = True

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        if self.training:
            self.set_batch_first(lprobs)
        return lprobs

    def output_layout(self):
        return "BTD"

    """
    The forward method inherited from the base class has a **kwargs argument in
    its input, which is not supported in torchscript. This method overrites the forward
    method definition without **kwargs.
    """

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class ConvTransformerEncoder(FairseqEncoder):
    """Conv + Transformer encoder"""

    def __init__(self, cfg):
        """Construct an Encoder object."""
        super().__init__(None)

        self.dropout = cfg.dropout
        self.embed_scale = (
            1.0 if cfg.no_scale_embedding else math.sqrt(cfg.encoder_embed_dim)
        )
        self.padding_idx = 1
        self.in_channels = 1
        self.input_dim = cfg.input_feat_per_channel
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, cfg.conv_out_channels, 3, stride=2, padding=3 // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                cfg.conv_out_channels,
                cfg.conv_out_channels,
                3,
                stride=2,
                padding=3 // 2,
            ),
            torch.nn.ReLU(),
        )
        transformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim, cfg.conv_out_channels
        )
        self.out = torch.nn.Linear(transformer_input_dim, cfg.encoder_embed_dim)
        self.embed_positions = PositionalEmbedding(
            cfg.max_source_positions,
            cfg.encoder_embed_dim,
            self.padding_idx,
            learned=False,
        )

        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend(
            [TransformerEncoderLayer(cfg) for i in range(cfg.encoder_layers)]
        )
        if cfg.encoder_normalize_before:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        else:
            self.layer_norm = None

    def pooling_ratio(self):
        return 4

    def infer_conv_output_dim(self, in_channels, input_dim, out_channels):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        x = torch.nn.Conv2d(1, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    def forward(self, src_tokens, src_lengths):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = (
            src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
            .transpose(1, 2)
            .contiguous()
        )
        x = self.conv(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        x = self.out(x)
        x = self.embed_scale * x

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        input_len_0 = (src_lengths.float() / subsampling_factor).ceil().long()
        input_len_1 = x.size(0) * torch.ones([src_lengths.size(0)]).long().to(input_len_0.device)
        input_lengths = torch.min(
            input_len_0, input_len_1
        )

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)

        if not encoder_padding_mask.any():
            maybe_encoder_padding_mask = None
        else:
            maybe_encoder_padding_mask = encoder_padding_mask


        return {
            "encoder_out": [x],
            "encoder_padding_mask": [maybe_encoder_padding_mask]
            if maybe_encoder_padding_mask is not None
            else [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                (encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                (encoder_out["encoder_embedding"][0]).index_select(0, new_order)
            ]
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": new_encoder_embedding,
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
        }


class TransformerDecoderNoExtra(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None
