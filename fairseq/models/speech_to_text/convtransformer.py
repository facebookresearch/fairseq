#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple

from fairseq import checkpoint_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer
)
from .vggtransformer import (
    lengths_to_encoder_padding_mask
)
from torch import Tensor


logger = logging.getLogger(__name__)


@register_model("convtransformer")
class ConvTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--max-source-positions",
            default=3000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument("--load-pretrained-encoder-from", type=str,
                            metavar="STR", help=(
                                "model to take encoder"
                                "weights from (for initialization)"))

    @classmethod
    def build_encoder(cls, args):
        encoder = ConvTransformerEncoder(args)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderNoExtra(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
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
        self.set_batch_first(lprobs)
        return lprobs

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

    def __init__(self, args):
        """Construct an Encoder object."""
        super().__init__(None)

        self.dropout = args.dropout
        self.embed_scale = (
            1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
        )
        self.padding_idx = 1
        self.in_channels = 1
        self.input_dim = args.input_feat_per_channel
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, args.encoder_embed_dim, 3, stride=2, padding=3 // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                args.encoder_embed_dim,
                args.encoder_embed_dim,
                3,
                stride=2,
                padding=3 // 2,
            ),
            torch.nn.ReLU(),
        )
        transformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim, args.encoder_embed_dim
        )
        self.out = torch.nn.Linear(transformer_input_dim, args.encoder_embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            args.encoder_embed_dim,
            self.padding_idx,
            learned=False,
        )

        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def infer_conv_output_dim(self, in_channels, input_dim, encoder_embed_dim):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        x = torch.nn.Conv2d(1, encoder_embed_dim, 3, stride=2, padding=3 // 2)(x)
        x = torch.nn.Conv2d(
            encoder_embed_dim, encoder_embed_dim, 3, stride=2, padding=3 // 2
        )(x)
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

        subsampling_factor = 1.0 * max_seq_len / output_seq_len
        input_lengths = (src_lengths.float() / subsampling_factor).round().long()

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )

        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)

        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)


        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)

        if not encoder_padding_mask.any():
            maybe_encoder_padding_mask = None
        else:
            maybe_encoder_padding_mask = encoder_padding_mask

        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=maybe_encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        """Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement"""

        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )

        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )

        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


class TransformerDecoderNoExtra(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
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


@register_model_architecture(model_name="convtransformer", arch_name="convtransformer")
def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.max_source_positions = getattr(args, "max_source_positions", 3000)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)


@register_model_architecture("convtransformer", "convtransformer_espnet")
def convtransformer_espnet(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
