# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture,
)
from fairseq.modules import LayerNorm, TransformerDecoderLayer

logger = logging.getLogger(__name__)


@register_model("laser_transformer")
class LaserTransformerModel(FairseqEncoderDecoderModel):
    """Train Transformer for LASER task

    Requires --task laser
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens=None,
        tgt_tokens=None,
        tgt_lengths=None,
        target_language_id=-1,
        dataset_name="",
    ):
        laser_encoder_out = self.encoder(src_tokens, src_lengths)
        return self.decoder(
            prev_output_tokens, laser_encoder_out, lang_id=target_language_id
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--decoder-lang-embed-dim",
            type=int,
            metavar="N",
            help="decoder language embedding dimension",
        )
        parser.add_argument(
            "--sentemb-criterion",
            choices=["maxpool", "cls"],
            help="How to build sentence embeddings?",
        )

    @classmethod
    def build_model(cls, args, task):
        base_laser_transformer_architecture(args)

        num_langs = task.num_tasks if hasattr(task, "num_tasks") else 0

        def load_embed_tokens(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()

            return Embedding(num_embeddings, embed_dim, padding_idx)

        encoder_embed_tokens = load_embed_tokens(
            task.source_dictionary, args.encoder_embed_dim
        )
        decoder_embed_tokens = load_embed_tokens(
            task.target_dictionary, args.decoder_embed_dim
        )
        num_langs = task.num_tasks if hasattr(task, "num_tasks") else 0

        if args.sentemb_criterion == "cls":
            assert args.prepend_bos, "With CLS sentemb criterion, use --prepend-bos"

        encoder = LaserTransformerEncoder(
            args.sentemb_criterion, args, task.source_dictionary, encoder_embed_tokens
        )

        decoder = LaserTransformerDecoder(
            args,
            task.target_dictionary,
            decoder_embed_tokens,
            num_langs=num_langs,
            lang_embed_dim=args.decoder_lang_embed_dim,
        )

        return cls(encoder, decoder)


class LaserTransformerEncoder(TransformerEncoder):
    def __init__(self, sentemb_criterion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sentemb_criterion = sentemb_criterion
        namespace = args[0]
        dictionary = args[1]
        tasks = [
            task.split(":")[0] for task in namespace.student_teacher_config.split(",")
        ]
        # if we have a masking task, then add a linear layer projecting from embed_dim to vocab_size
        if "mask" in tasks:
            self.project_vocabulary = nn.Linear(
                namespace.encoder_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.project_vocabulary.weight,
                mean=0,
                std=namespace.encoder_embed_dim**-0.5,
            )
            self.lm_head_transform_weight = nn.Linear(
                namespace.encoder_embed_dim, namespace.encoder_embed_dim
            )
            self.activation_fn = utils.get_activation_fn("tanh")
            self.layer_norm = LayerNorm(namespace.encoder_embed_dim)

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def forward(
        self,
        src_tokens,
        src_lengths,
        masked_tokens=None,
        prev_output_tokens=None,
        target_language_id=-1,
        dataset_name="",
    ):
        encoder_out = super().forward(src_tokens, src_lengths)

        x = encoder_out["encoder_out"][0]  # T x B x D

        # project masked tokens only if performing MLM task
        if isinstance(masked_tokens, torch.Tensor):
            x = x.transpose(0, 1)  # B x T x D
            # only project the masked tokens to size of vocabulary (other tokens are not considered for loss)
            x = x[masked_tokens, :]
            # project back to size of vocabulary
            x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
            x = self.project_vocabulary(x)  # B x V
            # return logits for mlm criterion
            return [x]  # MLM criterion takes first element of list as logits
        else:
            # if not MLM task return sentence embedding
            padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)

            if padding_mask.any() and self.sentemb_criterion == "maxpool":
                x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)

            if self.sentemb_criterion == "cls":
                # determine location of 'cls' e.g. due to left-padding 'cls' may be different each sent
                cls_indices = src_tokens.eq(self.dictionary.bos()).t()  # T x B
                sentemb = x[cls_indices, :]
            # Build the sentence embedding by max-pooling over the encoder outputs
            elif self.sentemb_criterion == "maxpool":
                sentemb = x.max(dim=0)[0]
            else:
                raise Exception(
                    "Please provide a sentence embedding option from [cls|maxpool]"
                )

            return {"sentemb": sentemb}  # B x D

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out_dict, new_order):
        encoder_out_dict["sentemb"] = encoder_out_dict["sentemb"].index_select(
            0, new_order
        )

        return encoder_out_dict


class LaserTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, *kargs, **kwargs):
        self.num_langs = kwargs.get("num_langs", 1)
        self.lang_embed_dim = kwargs.get("lang_embed_dim", 0)
        kwargs.pop("num_langs", None)
        kwargs.pop("lang_embed_dim", None)

        super().__init__(args, dictionary, *kargs, **kwargs, no_encoder_attn=True)

        if self.lang_embed_dim == 0:
            self.embed_lang = None
        else:
            self.embed_lang = nn.Embedding(self.num_langs, self.lang_embed_dim)
            nn.init.uniform_(self.embed_lang.weight, -0.1, 0.1)

        if self.output_projection is not None:
            laser_output_embed_dim = (
                self.output_embed_dim + self.lang_embed_dim + args.encoder_embed_dim
            )
            self.output_projection = nn.Linear(
                laser_output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight,
                mean=0,
                std=laser_output_embed_dim**-0.5,
            )

    def build_decoder_layer(self, args, no_encoder_attn=False, is_moe_layer=False):
        decoder_embed_dim = args.decoder_embed_dim
        args.decoder_embed_dim = (
            decoder_embed_dim + self.lang_embed_dim + args.encoder_embed_dim
        )
        res = TransformerDecoderLayer(args, no_encoder_attn=True)
        args.decoder_embed_dim = decoder_embed_dim

        return res

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        lang_id: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

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

        if self.embed_lang is not None:
            lang_ids = prev_output_tokens.data.new_full((bsz,), lang_id)
            langemb = self.embed_lang(lang_ids)
            langemb = langemb.unsqueeze(0)
            repeat_vals = [x.shape[0] // langemb.shape[0]] + [-1] * (
                len(langemb.shape) - 1
            )
            x = torch.cat((x, langemb.expand(*repeat_vals)), dim=-1)

        sentemb = encoder_out["sentemb"]
        sentemb = sentemb.unsqueeze(0)

        repeat_vals = [x.shape[0] // sentemb.shape[0]] + [-1] * (len(sentemb.shape) - 1)
        x = torch.cat((x, sentemb.expand(*repeat_vals)), dim=-1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(
                    x.transpose(0, 1)
                )  # `buffered_future_mask` is expecting B x T x C
            else:
                self_attn_mask = None

            x, layer_attn, _, l_aux = layer(
                x,
                None,
                None,
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

        return x, {"attn": [attn], "inner_states": inner_states}

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        lang_id: Optional[int] = None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        assert lang_id is not None

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            lang_id=lang_id,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra


@register_model_architecture("laser_transformer", "laser_transformer")
def base_laser_transformer_architecture(args):
    base_architecture(args)
    args.decoder_lang_embed_dim = getattr(args, "decoder_lang_embed_dim", 0)
