# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_model import FairseqEncoderModel
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.modules.transformer_layer import TransformerEncoderLayer

from .s2t_dualinputwavtransformer import DualInputWavTransformerModel, dualinputs2twavtransformer_base


class EntityRetrievalLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.embed_dim = args.er_encoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.er_dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=args.activation_fn)
        activation_dropout_p = args.er_activation_dropout
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.attn = self.build_retrieval_attention(args)
        self.attn_layer_norm = LayerNorm(self.embed_dim)
        self.normalize_before = args.er_encoder_normalize_before

        self.fc1 = nn.Linear(
            self.embed_dim,
            args.er_encoder_ffn_embed_dim,
        )
        self.fc2 = nn.Linear(
            args.er_encoder_ffn_embed_dim,
            self.embed_dim,
        )
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        text_x,
        speech_x,
        text_padding_mask=None,
        speech_padding_mask=None,
    ):
        """
        Args:
            text_x (Tensor): input to the layer of shape `(text_len, batch, embed_dim)`
            speech_x (Tensor): input to the layer of shape `(speech_len, batch, embed_dim)`
            text_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, text_len)` where padding
                elements are indicated by ``1``.
            speech_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, speech_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(speech_len, batch, embed_dim)`
        """
        x = text_x
        if self.normalize_before:
            x = self.attn_layer_norm(x)
        x.masked_fill_(text_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        x, attn = self.attn(
            query=x,
            key=speech_x,
            value=speech_x,
            key_padding_mask=speech_padding_mask,
            static_kv=True,
        )
        x.masked_fill_(text_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        x = self.dropout_module(x)
        if not self.normalize_before:
            x = self.attn_layer_norm(x)

        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn

    @classmethod
    def build_retrieval_attention(cls, args):
        return MultiheadAttention(
            args.er_encoder_embed_dim,
            args.er_encoder_attention_heads,
            kdim=args.er_encoder_embed_dim,
            vdim=args.er_encoder_embed_dim,
            dropout=args.er_attention_dropout,
            encoder_decoder_attention=True,
        )


class EntityRetrievalNetwork(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.init_proj = None
        if args.er_encoder_embed_dim != args.encoder_embed_dim:
            self.init_proj = nn.Linear(args.encoder_embed_dim, args.er_encoder_embed_dim, bias=False)
        self.layers = nn.ModuleList([
            EntityRetrievalLayer(args) for _ in range(args.er_encoder_layers)])
        self.output_proj = nn.Linear(args.er_encoder_embed_dim, 1, bias=False)

    @staticmethod
    def padding_aware_mean(x, padding_mask, batch_dim=1, time_dim=0, channel_dim=2):
        lengths = (1 - padding_mask.long()).sum(dim=-1)
        weight_matrix = torch.zeros(x.shape[batch_dim], x.shape[time_dim])
        for b in range(x.shape[batch_dim]):
            weight_matrix[b, :lengths[b]] = 1.0 / lengths[b].float()
        return x.permute(batch_dim, channel_dim, time_dim).bmm(weight_matrix.to(x.device).unsqueeze(-1)).squeeze()

    def forward(self, text_encoded, speech_encoded, text_padding_mask, speech_padding_mask):
        x = text_encoded
        for layer in self.layers:
            x, _ = layer(x, speech_encoded, text_padding_mask, speech_padding_mask)
        # average over time dim, and then project from (B, C) to (B, 1)
        return self.output_proj(self.padding_aware_mean(x, text_padding_mask))


class ERTransformerEncoderLayer(TransformerEncoderLayer):
    def build_self_attention(self, embed_dim, args):
        cfg = TransformerConfig.from_namespace(args)
        attn_activation_fn = getattr(args, 'er_activation_fn', 'softmax')
        if attn_activation_fn == 'softmax':
            attn_activation_fn = None  # this is the default in multihead attention
        elif attn_activation_fn in {'entmax15', 'sparsemax'}:
            import entmax
            attn_activation_fn = getattr(entmax, attn_activation_fn)
        else:
            raise Exception(
                f"Unexpected argument {attn_activation_fn} for self-attention activation funciton")
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
            attn_activation_fn=attn_activation_fn,
        )


class ERBertBased(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.cls_vector = nn.Parameter(torch.zeros(args.encoder_embed_dim))
        self.sep_vector = nn.Parameter(torch.zeros(args.encoder_embed_dim))
        self.layers = nn.ModuleList([
            ERTransformerEncoderLayer(args) for _ in range(args.er_encoder_layers)])
        self.output_proj = nn.Linear(args.er_encoder_embed_dim, 1, bias=False)

    def forward(self, text_encoded, speech_encoded, text_padding_mask, speech_padding_mask):
        speech_lengths = (1 - speech_padding_mask.long()).sum(dim=-1)
        text_lengths = (1 - text_padding_mask.long()).sum(dim=-1)
        concat_elements = []
        new_lens = []
        for b in range(text_encoded.shape[1]):
            concat_elem = torch.cat([
                self.cls_vector.unsqueeze(0), text_encoded[:text_lengths[b], b, :],
                self.sep_vector.unsqueeze(0), speech_encoded[:speech_lengths[b], b, :]])
            new_lens.append(concat_elem.shape[0])
            concat_elements.append(concat_elem)
        max_len = max(new_lens)
        for i, elem in enumerate(concat_elements):
            if elem.shape[0] < max_len:
                concat_elements[i] = torch.cat([
                    elem, torch.zeros(max_len - elem.shape[0], elem.shape[1]).to(elem.device)
                ])
        x = torch.stack(concat_elements).transpose(0, 1)
        padding_mask = lengths_to_padding_mask(torch.tensor(new_lens)).to(x.device)
        for layer in self.layers:
            x = layer(x, padding_mask)
        # take CLS token over time, and then project from (B, C) to (B, 1)
        return self.output_proj(x[0, :, :])


@register_model("dual_input_er_transformer")
class EntityRetrievalModel(FairseqEncoderModel):
    def __init__(self, encoder, retrieval_network):
        super().__init__(encoder)
        self.retrieval_network = retrieval_network

    @staticmethod
    def add_args(parser):
        DualInputWavTransformerModel.add_args(parser)
        parser.add_argument(
            "--er-dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--er-attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--er-activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--er-encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--er-encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--er-encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--er-encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--er-encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--pretrained-encoder",
            type=str,
            metavar="PATH",
            help="path where to retrieve pretrained encoders",
        )
        parser.add_argument(
            "--retrieval-network",
            type=str,
            metavar="NET",
            choices=['bert_like', 'speech2slot'],
            help="type of retrieval network to use",
        )
        parser.add_argument(
            "--er-activation-fn",
            type=str,
            metavar="FN",
            choices=['softmax', 'entmax15', 'sparsemax'],
            help="type of retrieval network to use",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        encoder = cls.build_encoder(args, task)
        if args.retrieval_network == "bert_like":
            retrieval_network = ERBertBased(args)
        elif args.retrieval_network == "speech2slot":
            retrieval_network = EntityRetrievalNetwork(args)
        else:
            raise Exception(f"Entity retrieval network {args.retrieval_network} not supported")
        return cls(encoder, retrieval_network)

    @classmethod
    def build_encoder(cls, args, task):
        encoder = DualInputWavTransformerModel.build_encoder(args, task)
        if getattr(args, "pretrained_encoder", None) is not None:
            encoder = checkpoint_utils.load_pretrained_component_from_model(encoder, args.pretrained_encoder)
        # Freeze encoder parameters
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder

    def forward(
        self,
        src_tokens,
        src_lengths,
        positive_retr_tokens,
        negative_retr_tokens,
        **kwargs
    ):
        with torch.no_grad():
            speech_encoder_out = self.encoder.spch_encoder(src_tokens, src_lengths, return_all_hiddens=False)
            positive_encoder_outs = [
                self.encoder.text_encoder(
                    pr_tokens, pr_tokens.ne(self.encoder.dictionary.pad()).sum(dim=1).long()
                ) for pr_tokens in positive_retr_tokens
            ]
            negative_encoder_outs = [
                self.encoder.text_encoder(
                    nr_tokens, nr_tokens.ne(self.encoder.dictionary.pad()).sum(dim=1).long()
                ) for nr_tokens in negative_retr_tokens
            ]
        positive_retrieval_encoder_out = [
            self.retrieval_network(
                p_encoded['encoder_out'][0],
                speech_encoder_out['encoder_out'][0],
                p_encoded['encoder_padding_mask'][0],
                speech_encoder_out['encoder_padding_mask'][0])
            for p_encoded in positive_encoder_outs
        ]
        negative_retrieval_encoder_out = [
            self.retrieval_network(
                n_encoded['encoder_out'][0],
                speech_encoder_out['encoder_out'][0],
                n_encoded['encoder_padding_mask'][0],
                speech_encoder_out['encoder_padding_mask'][0])
            for n_encoded in negative_encoder_outs
        ]
        return {
            "encoder_out": [speech_encoder_out],
            "positive_retrieval_out": positive_retrieval_encoder_out,
            "negative_retrieval_out": negative_retrieval_encoder_out,
        }


@register_model_architecture(
    "dual_input_er_transformer", "dual_input_er_transformer_base"
)
def dual_input_er_transformer_base(args):
    dualinputs2twavtransformer_base(args)
    args.er_encoder_layers = getattr(args, "er_encoder_layers", 3)
    args.er_encoder_normalize_before = getattr(args, "er_encoder_normalize_before", args.encoder_normalize_before)
    args.er_encoder_attention_heads = getattr(args, "er_encoder_attention_heads", args.encoder_attention_heads)
    args.er_encoder_embed_dim = getattr(args, "er_encoder_embed_dim", args.encoder_embed_dim)
    args.er_encoder_ffn_embed_dim = getattr(
        args, "er_encoder_ffn_embed_dim", args.er_encoder_embed_dim * 4
    )
    args.er_dropout = getattr(args, "er_dropout", args.dropout)
    args.er_attention_dropout = getattr(args, "er_attention_dropout", args.attention_dropout)
    args.er_activation_dropout = getattr(args, "er_activation_dropout", args.er_dropout)
    args.retrieval_network = getattr(args, "retrieval_network", "bert_like")


@register_model_architecture(
    "dual_input_er_transformer", "dual_input_er_transformer_base_speech2slot"
)
def dual_input_er_transformer_base_speech2slot(args):
    args.retrieval_network = getattr(args, "retrieval_network", "speech2slot")
    dualinputs2twavtransformer_base(args)


@register_model_architecture(
    "dual_input_er_transformer", "dual_input_er_transformer_base_stack"
)
def dual_input_er_transformer_base_stack(args):
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 6)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.encoder_shared_text_layers_from_begin = getattr(
        args, "encoder_shared_text_layers_from_begin", 0
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.stacked_encoder = getattr(args, "stacked_encoder", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    dual_input_er_transformer_base(args)


@register_model_architecture(
    "dual_input_er_transformer", "dual_input_er_transformer_large"
)
def dual_input_er_transformer_large(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 24)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 12)
    args.encoder_shared_text_layers_from_begin = getattr(
        args, "encoder_shared_text_layers_from_begin", 12
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    dual_input_er_transformer_base(args)
