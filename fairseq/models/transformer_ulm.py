# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from fairseq.models.fairseq_decoder import FairseqDecoder
import numpy as np
from typing import Optional, Dict, Any, List
import torch
from torch import nn
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.tasks.speech_ulm_task import SpeechUnitLanguageModelingTask
from fairseq.models.transformer import Embedding, TransformerDecoder, Linear
from fairseq.models.transformer_lm import TransformerLanguageModelConfig
from torch import Tensor


DEFAULT_MAX_TARGET_POSITIONS = 1024
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class SpeechUnitLanguageModelConfig(TransformerLanguageModelConfig):
    mask_unit_seg_prob: float = field(
        default=0.0, metadata={"help": "probability to mask a segment of unit sequence"}
    )
    mask_unit_seg_leng: int = field(
        default=5, metadata={"help": "length of unit segment mask"}
    )
    mask_unit_seg_type: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose unit mask length"}
    )

    mask_dur_prob: float = field(
        default=0.0, metadata={"help": "probability to mask entire duration sequence"}
    )
    mask_dur_seg_prob: float = field(
        default=0.0,
        metadata={"help": "probability to mask a segment of duration sequence"},
    )
    mask_dur_seg_leng: int = field(
        default=5, metadata={"help": "length of duration segment mask"}
    )
    mask_dur_seg_type: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose duration mask length"}
    )

    mask_f0_prob: float = field(
        default=0.0, metadata={"help": "probability to mask entire duration sequence"}
    )
    mask_f0_seg_prob: float = field(
        default=0.0, metadata={"help": "probability to mask a segment of f0 sequence"}
    )
    mask_f0_seg_leng: int = field(
        default=5, metadata={"help": "length of f0 segment mask"}
    )
    mask_f0_seg_type: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose f0 mask length"}
    )


@register_model("transformer_ulm", dataclass=SpeechUnitLanguageModelConfig)
class TransformerUnitLanguageModel(FairseqLanguageModel):
    def __init__(
        self,
        cfg: SpeechUnitLanguageModelConfig,
        task: SpeechUnitLanguageModelingTask,
        decoder: FairseqDecoder,
    ):
        super().__init__(decoder)
        self.cfg = cfg

        self.channel_names = task.channel_names
        self.channel_sizes = task.channel_sizes

        self.unit_mask_val = task.source_dictionary.unk()
        self.dur_mask_val = (
            task.source_duration_dictionary.unk() if task.cfg.discrete_duration else 0
        )
        self.f0_mask_val = (
            task.source_f0_dictionary.unk() if task.cfg.discrete_f0 else 0
        )

        self.ignore_duration_input = task.cfg.ignore_duration_input
        self.ignore_f0_input = task.cfg.ignore_f0_input

    @classmethod
    def build_model(cls, args, task):
        base_ulm_architecture(args)

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = Embedding(
            len(task.source_dictionary),
            args.decoder_input_dim,
            padding_idx=task.source_dictionary.pad(),
        )
        embed_duration = None
        if task.cfg.discrete_duration:
            embed_duration = Embedding(
                len(task.source_duration_dictionary),
                args.decoder_input_dim,
                padding_idx=0,  # duration uses 0 for padding
            )
        embed_f0 = None
        if task.cfg.discrete_f0:
            embed_f0 = Embedding(
                len(task.source_f0_dictionary),
                args.decoder_input_dim,
                padding_idx=task.source_f0_dictionary.pad(),
            )

        decoder = MultiStreamTransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            [embed_duration, embed_f0],
            no_encoder_attn=True,
            channel_sizes=task.channel_sizes,
        )

        return cls(args, task, decoder)

    def apply_seg_dropout(self, inp, mask_prob, mask_leng, mask_type, mask_val):
        B, T = inp.size()
        if mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T), None, mask_prob, mask_leng, mask_type  # may mask padding
            )
            mask_indices = torch.from_numpy(mask_indices).to(inp.device)
            inp[mask_indices] = mask_val
        else:
            mask_indices = torch.zeros_like(inp).bool()
        return inp, mask_indices

    def apply_seq_dropout(self, inp, mask_prob, mask_val):
        B, T = inp.size()
        if mask_prob > 0:
            mask_indices = np.random.uniform(0, 1, (B,)) < mask_prob
            mask_indices = (
                torch.from_numpy(mask_indices).to(inp.device).unsqueeze(1).expand(-1, T)
            )
            inp[mask_indices] = mask_val
        else:
            mask_indices = torch.zeros_like(inp).bool()
        return inp, mask_indices

    def apply_dropout(self, src_tokens, dur_src, f0_src):
        src_tokens, unit_mask = self.apply_seg_dropout(
            src_tokens,
            self.cfg.mask_unit_seg_prob,
            self.cfg.mask_unit_seg_leng,
            self.cfg.mask_unit_seg_type,
            self.unit_mask_val,
        )

        dur_src, dur_mask = self.apply_seq_dropout(
            dur_src, self.cfg.mask_dur_prob, self.dur_mask_val
        )
        dur_src, _dur_mask = self.apply_seg_dropout(
            dur_src,
            self.cfg.mask_dur_seg_prob,
            self.cfg.mask_dur_seg_leng,
            self.cfg.mask_dur_seg_type,
            self.dur_mask_val,
        )
        dur_mask = dur_mask.logical_or(_dur_mask)

        f0_src, f0_mask = self.apply_seq_dropout(
            f0_src, self.cfg.mask_f0_prob, self.f0_mask_val
        )
        f0_src, _f0_mask = self.apply_seg_dropout(
            f0_src,
            self.cfg.mask_f0_seg_prob,
            self.cfg.mask_f0_seg_leng,
            self.cfg.mask_f0_seg_type,
            self.f0_mask_val,
        )
        f0_mask = f0_mask.logical_or(_f0_mask)

        return src_tokens, unit_mask, dur_src, dur_mask, f0_src, f0_mask

    def forward(
        self,
        src_tokens: torch.Tensor,
        dur_src: torch.Tensor,
        f0_src: torch.Tensor,
        src_lengths: Optional[Any] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        if self.ignore_duration_input:
            dur_src = torch.zeros_like(dur_src)

        if self.ignore_f0_input:
            f0_src = torch.zeros_like(f0_src)

        if self.training:
            (
                src_tokens,
                unit_mask,
                dur_src,
                dur_mask,
                f0_src,
                f0_mask,
            ) = self.apply_dropout(src_tokens, dur_src, f0_src)
        else:
            unit_masks = dur_mask = f0_mask = None

        prediction, _ = self.decoder(
            prev_output_tokens=(src_tokens, dur_src, f0_src),
            incremental_state=incremental_state,
            src_lengths=src_lengths,
            features_only=True,
        )

        result = dict(zip(self.channel_names, prediction))

        return result


def base_ulm_architecture(args):
    from .transformer_lm import base_lm_architecture

    base_lm_architecture(args)


@register_model_architecture("transformer_ulm", "transformer_ulm_big")
def transformer_ulm_big(args):
    from .transformer_lm import transformer_lm_big

    transformer_lm_big(args)
    base_ulm_architecture(args)


@register_model_architecture("transformer_ulm", "transformer_ulm_tiny")
def transformer_ulm_tiny(args):
    from .transformer_lm import transformer_lm_gpt2_tiny

    transformer_lm_gpt2_tiny(args)
    base_ulm_architecture(args)


class MultiStreamTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        embed_other_list,
        no_encoder_attn,
        channel_sizes,
    ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )

        # embed each channel and project if dimensions do not match
        self.embed_other_list = torch.nn.ModuleList(embed_other_list)
        self.proj_other_list = torch.nn.ModuleList()
        dim = embed_tokens.embedding_dim
        for embed_other in embed_other_list:
            other_dim = 1 if embed_other is None else embed_other.embedding_dim
            self.proj_other_list.append(
                nn.Linear(other_dim, dim) if other_dim != dim else None
            )

        # tranformer output to prediction
        self.channel_sizes = channel_sizes
        self.project_out_dim = Linear(
            embed_tokens.embedding_dim, sum(channel_sizes), bias=False
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # XXX: first multi-channel change start
        prev_output_tokens, *other_channels = prev_output_tokens
        # XXX: first multi-channel change end

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            other_channels = [o[:, -1:] for o in other_channels]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        # XXX: second multi-channel change start
        other_channels = [
            o.unsqueeze(-1).to(dtype=x.dtype) if emb is None else emb(o)
            for o, emb in zip(other_channels, self.embed_other_list)
        ]
        other_channels = [
            o if proj_other is None else proj_other(o)
            for o, proj_other in zip(other_channels, self.proj_other_list)
        ]
        for o in other_channels:
            x = x + o
        # XXX: second multi-channel change end

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        else:
            assert False

        # XXX: the last change start
        result = []
        start = 0
        for channel_size in self.channel_sizes:
            end = start + channel_size
            result.append(x[:, :, start:end])
            start = end
        assert end == x.size(-1)
        # XXX: the last change end

        return result, {"attn": [attn], "inner_states": inner_states}
