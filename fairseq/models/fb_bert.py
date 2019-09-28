# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (
    FairseqDecoder, BaseFairseqModel, register_model, register_model_architecture,
)

from fairseq.models.transformer import PositionalEmbedding

from fairseq.modules import (
    SinusoidalPositionalEmbedding, MultiheadAttention,
)


def gelu(x):
    """Implementation of the gelu activation function
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@register_model('masked_lm')
class MaskedLMModel(BaseFairseqModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """
    def __init__(self, args, decoder):
        super().__init__()
        self.args = args
        self.decoder = decoder
        self.apply(self.init_masked_lm_weights)

    def init_masked_lm_weights(self, module):
        """
        Initialize the weights. This overrides the default initializations
        depending on the specified arguments.
            1. If normal-init-lin-weights is set then weights of linear
               layer will be initialized using the normal distribution and
               bais will be set to the specified value.
            2. If normal-init-embed-weights is set then weights of embedding
               layer will be initialized using the normal distribution.
            3. If we use the custom BertLayerNorm, weights will be init
               with constant value (1.0 by default).
            4. If normal-init-proj-weights s set then weights of
               in_project_weight for MultiHeadAttention initialized using
               the normal distribution (to be validated).
        """
        if isinstance(module, nn.Linear) and \
                self.args.normal_init_lin_weights:
            module.weight.data.normal_(
                mean=self.args.init_lin_weight_mean,
                std=self.args.init_lin_weight_std
            )
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding) and \
                self.args.normal_init_embed_weights:
            module.weight.data.normal_(
                mean=self.args.init_embed_weight_mean,
                std=self.args.init_embed_weight_std
            )
        if isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.args.layernorm_init_val)
        if isinstance(module, MultiheadAttention) and \
                self.args.normal_init_proj_weights:
            module.in_proj_weight.data.normal_(
                mean=self.args.init_proj_weight_mean,
                std=self.args.init_proj_weight_std
            )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', default=0.1, type=float,
                            metavar='D', help='dropout probability for'
                            ' attention weights')
        parser.add_argument('--act-dropout', default=0.1, type=float,
                            metavar='D', help='dropout probability after'
                            ' activation in FFN')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--no-bias-kv', action='store_true',
                            help='if set, pads attn with zero instead of'
                            ' adding a learnable bias kv')

        # Arguments related to input and output embeddings
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--share-decoder-input-output-embed',
                            action='store_true', help='share decoder input'
                            ' and output embeddings')
        parser.add_argument('--no-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings'
                            ' (outside self attention)')
        parser.add_argument('--num-segment', type=int, metavar='N', default=2,
                            help='num segment in the input')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N',
                            default=2, help='number of classes for sentence'
                            ' task')
        parser.add_argument('--sent-loss', action='store_true', help='if set,'
                            ' calculate sentence level predictions')

        # Arguments related to parameter initialization
        # linear layers
        parser.add_argument('--normal-init-lin-weights', action='store_true',
                            help='initialize weights of linear layers using'
                            ' normal distribution')
        parser.add_argument('--init-lin-weight-std', default=1.0, type=float,
                            metavar='D', help='std for initializing linear'
                            ' layers using normal distribution')
        parser.add_argument('--init-lin-weight-mean', default=0.0, type=float,
                            metavar='D', help='mean for initializing linear'
                            ' layers using normal distribution')
        parser.add_argument('--init-lin-bias', default=0.0, type=float,
                            metavar='D', help='initialize the bias for the'
                            ' linear layer with the specified value')
        # embedding layers
        parser.add_argument('--normal-init-embed-weights', action='store_true',
                            help='initialize weights of embedding layers using'
                            ' normal distribution')
        parser.add_argument('--init-embed-weight-std', default=1.0, type=float,
                            metavar='D', help='std for initializing embedding'
                            ' layers using normal distribution')
        parser.add_argument('--init-embed-weight-mean', default=0.0, type=float,
                            metavar='D', help='mean for initializing embedding'
                            ' layers using normal distribution')
        # layer norm layers
        parser.add_argument('--bert-layer-norm', action='store_true',
                            help='use custom Layer Norm module for BERT')
        parser.add_argument('--layernorm-init-val', default=1.0, type=float,
                            metavar='D', help='init value for weights of'
                            ' layer norm')
        # multi-head attention
        parser.add_argument('--normal-init-proj-weights', action='store_true',
                            help='initialize weights of embedding layers using'
                            ' normal distribution')
        parser.add_argument('--init-proj-weight-std', default=1.0, type=float,
                            metavar='D', help='std for initializing embedding'
                            ' layers using normal distribution')
        parser.add_argument('--init-proj-weight-mean', default=0.0, type=float,
                            metavar='D', help='mean for initializing embedding'
                            ' layers using normal distribution')

        # misc params
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--gelu', action='store_true',
                            help='Use gelu activation function in Decoder'
                            ' Layer')

    def forward(self, src_tokens, segment_labels, **unused):
        return self.decoder(src_tokens, segment_labels)

    def max_positions(self):
        return self.decoder.max_positions

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if args.task == 'bert':
            base_bert_architecture(args)
        else:
            xlm_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        print("Model args: ", args)

        decoder = MaskedLMDecoder(args, task.dictionary)
        return cls(args, decoder)


class MaskedLMDecoder(FairseqDecoder):
    """
    Decoder for Masked Language Modelling.
    """

    def __init__(self, args, dictionary, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = args.decoder_embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), embed_dim, self.padding_idx
        )
        self.max_positions = args.max_positions

        self.embed_segment = nn.Embedding(
            args.num_segment, embed_dim, self.padding_idx,
        ) if args.num_segment > 0 else None

        self.embed_positions = PositionalEmbedding(
            self.max_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([MaskedLMDecoderLayer(args)
                                     for _ in range(args.decoder_layers)])

        self.load_softmax = not getattr(args, 'remove_head', False)
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num

        if self.load_softmax:
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    embed_dim, len(dictionary), bias=False)
            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(
                    embed_dim, self.sentence_out_dim, bias=False)

    def forward(self, tokens, segment_labels, **unused):
        """
        Forward pass for Masked LM decoder. This first computes the token
        embedding using the token embedding matrix, position embeddings (if
        specified) and segment embeddings (if specified). After applying the
        specified number of MaskedLMDecoderLayers, it creates the output dict.
        Here we assume that the sentence representation corresponds to the
        output of the classification_token (see bert_task or cross_lingual_lm
        task for more details).
        Args:
            - tokens: B x T matrix representing sentences
            - segment_labels: B x T matrix representing segment label for tokens
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in
                  softmax afterwards
                - a dictionary of additional data, where 'sentence_rep' contains
                  the representation for classification_token and 'inner_states'
                  is a list of internal model states used to compute the
                  predictions (similar in ELMO). 'sentence_logits'
                  is the prediction logit for NSP task and is only computed if
                  this is specified in the input arguments.
        """
        # compute padding mask
        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        # embed positions
        positions = self.embed_positions(tokens) if self.embed_positions is not None else None

        # embed segments
        segment_embeddings = (
            self.embed_segment(segment_labels.long())
            if self.embed_segment is not None
            else None
        )

        # embed tokens, positions and segments
        x = self.embed_tokens(tokens)
        if positions is not None:
            x += positions
        if segment_embeddings is not None:
            x += segment_embeddings
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        #inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
            )
            #inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]

        # project back to size of vocabulary
        if not self.load_softmax:
            return x, sentence_rep
        if self.share_input_output_embed and hasattr(self.embed_tokens, 'weight'):
            x = F.linear(x, self.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        sentence_logits = None
        if self.sentence_projection_layer:
            sentence_logits = self.sentence_projection_layer(sentence_rep)

        return x, sentence_logits
            #'inner_states': inner_states,
            #'sentence_rep': sentence_rep,
            #'sentence_logits': sentence_logits
        #}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            state_dict[name + '.embed_positions._float_tensor'] = torch.FloatTensor(1)
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "sentence_projection_layer.weight" in k:
                    del state_dict[k]
        return state_dict


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style
        (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MaskedLMDecoderLayer(nn.Module):
    """
    Masked LM Decoder layer block. If the flag bert_layer_norm is set then
    we use the custom BertLayerNorm module instead of nn.LayerNorm.
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=not args.no_bias_kv,
            add_zero_attn=args.no_bias_kv,
        )
        self.dropout = args.dropout
        self.act_dropout = args.act_dropout
        self.normalize_before = args.decoder_normalize_before
        self.gelu = args.gelu

        self.self_attn_layer_norm = (
            BertLayerNorm(self.embed_dim) if args.bert_layer_norm
            else nn.LayerNorm(self.embed_dim, eps=1e-12)
        )

        self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = (
            BertLayerNorm(self.embed_dim) if args.bert_layer_norm
            else nn.LayerNorm(self.embed_dim, eps=1e-12)
        )

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = gelu(self.fc1(x)) if self.gelu else F.relu(self.fc1(x))
        x = F.dropout(x, p=self.act_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


@register_model_architecture('masked_lm', 'bert_base')
def base_bert_architecture(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', True)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.num_segment = getattr(args, 'num_segment', 2)

    args.decoder_layers = getattr(args, 'decoder_layers', 12)

    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.no_bias_kv = getattr(args, 'no_bias_kv', True)

    args.sent_loss = getattr(args, 'sent_loss', True)
    args.sentence_class_num = getattr(args, 'sentence-class-num', 2)

    args.normal_init_lin_weights = getattr(
        args, 'normal_init_lin_weights', True)
    args.init_lin_weight_std = 0.02
    args.init_lin_weight_mean = 0


    args.normal_init_embed_weights = getattr(
        args, 'normal_init_embed_weights', True)
    args.init_embed_weight_std = 0.02
    args.init_embed_weight_mean = 0

    args.normal_init_proj_weights = getattr(
        args, 'normal_init_proj_weights', True)
    args.init_proj_weight_std = 0.02
    args.init_proj_weight_mean = 0

    # TODO: validate setups for layernorm
    args.decoder_normalize_before = getattr(
        args, 'decoder_normalize_before', True)
    args.bert_layer_norm = getattr(args, 'bert_layer_norm', True)
    args.gelu = getattr(args, 'gelu', True)


@register_model_architecture('masked_lm', 'xlm_base')
def xlm_architecture(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', True)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.num_segment = getattr(args, 'num_segment', 1)

    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.no_bias_kv = getattr(args, 'no_bias_kv', True)

    args.normal_init_lin_weights = getattr(
        args, 'normal_init_lin_weights', False)
    args.normal_init_proj_weights = getattr(
        args, 'normal_init_proj_weights', False)

    args.normal_init_embed_weights = getattr(
        args, 'normal_init_embed_weights', True)
    args.init_embed_weight_std = 1024 ** -0.5

    args.sent_loss = getattr(args, 'sent_loss', False)

    # TODO: validate setups for layernorm
    args.decoder_normalize_before = getattr(
        args, 'decoder_normalize_before', False)
    args.bert_layer_norm = getattr(args, 'bert_layer_norm', False)
    args.gelu = getattr(args, 'gelu', True)

