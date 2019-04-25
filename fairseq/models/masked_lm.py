# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (
    BaseFairseqModel, FairseqEncoder, register_model, register_model_architecture,
)
from fairseq.modules import (
    SinusoidalPositionalEmbedding,
    TransformerSentenceEncoder
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params


@register_model('masked_lm')
class MaskedLMModel(BaseFairseqModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)

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
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--no-bias-kv', action='store_true',
                            help='if set, pads attn with zero instead of'
                            ' adding a learnable bias kv')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed',
                            action='store_true', help='share encoder input'
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
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')

        # layer norm layers
        parser.add_argument('--bert-layer-norm', action='store_true',
                            help='use custom Layer Norm module for BERT')

        # misc params
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--gelu', action='store_true',
                            help='Use gelu activation function in encoder'
                            ' Layer')

    def forward(self, tokens, segment_labels):
        return self.encoder(tokens, segment_labels)

    def max_positions(self):
        return self.encoder.max_positions

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

        encoder = MaskedLMEncoder(args, task.dictionary)
        return cls(args, encoder)


class MaskedLMEncoder(FairseqEncoder):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.max_positions = args.max_positions

        use_position_embeddings = (
            not getattr(args, 'no_token_positional_embeddings', False)
        )
        encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
        use_bert_layer_norm = getattr(args, 'bert_layer_norm', False)
        use_gelu = getattr(args, 'gelu', False)
        apply_bert_init = getattr(args, 'apply_bert_init', False)

        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=use_position_embeddings,
            encoder_normalize_before=encoder_normalize_before,
            use_bert_layer_norm=use_bert_layer_norm,
            use_gelu=use_gelu,
            apply_bert_init=apply_bert_init,
        )

        self.share_input_output_embed = getattr(
            args, 'share_encoder_input_output_embed', False)
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, 'remove_head', False)

        if self.load_softmax:
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim,
                    self.vocab_size,
                    bias=False
                )

            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(
                    args.encoder_embed_dim,
                    self.sentence_out_dim,
                    bias=False
                )

    def forward(self, tokens, segment_labels, **unused):
        """
        Forward pass for Masked LM encoder. This first computes the token
        embedding using the token embedding matrix, position embeddings (if
        specified) and segment embeddings (if specified).

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

        inner_states, sentence_rep = self.sentence_encoder(tokens, segment_labels)
        x = inner_states[-1].transpose(0, 1)

        # project back to size of vocabulary
        if self.share_input_output_embed \
                and hasattr(self.sentence_encoder.embed_tokens, 'weight'):
            x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        sentence_logits = None
        if self.sentence_projection_layer:
            sentence_logits = self.sentence_projection_layer(sentence_rep)

        return x, {
            'inner_states': inner_states,
            'sentence_rep': sentence_rep,
            'sentence_logits': sentence_logits
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(
                self.sentence_encoder.position_embeddings,
                SinusoidalPositionalEmbedding
        ):
            state_dict[
                name + '.sentence_encoder.position_embeddings._float_tensor'
            ] = torch.FloatTensor(1)
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "sentence_projection_layer.weight" in k:
                    del state_dict[k]
        return state_dict


@register_model_architecture('masked_lm', 'bert_base')
def base_bert_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.share_encoder_input_output_embed = getattr(
        args, 'share_encoder_input_output_embed', True)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.num_segment = getattr(args, 'num_segment', 2)

    args.encoder_layers = getattr(args, 'encoder_layers', 12)

    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.no_bias_kv = getattr(args, 'no_bias_kv', True)

    args.sent_loss = getattr(args, 'sent_loss', True)
    args.sentence_class_num = getattr(args, 'sentence-class-num', 2)

    args.apply_bert_init = getattr(args, 'apply_bert_init', True)

    # TODO: validate setups for layernorm
    args.encoder_normalize_before = getattr(
        args, 'encoder_normalize_before', True)
    args.bert_layer_norm = getattr(args, 'bert_layer_norm', True)
    args.gelu = getattr(args, 'gelu', True)


@register_model_architecture('masked_lm', 'xlm_base')
def xlm_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.share_encoder_input_output_embed = getattr(
        args, 'share_encoder_input_output_embed', True)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.num_segment = getattr(args, 'num_segment', 1)

    args.encoder_layers = getattr(args, 'encoder_layers', 6)

    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.no_bias_kv = getattr(args, 'no_bias_kv', True)

    args.sent_loss = getattr(args, 'sent_loss', False)

    args.encoder_normalize_before = getattr(
        args, 'encoder_normalize_before', False)
    args.bert_layer_norm = getattr(args, 'bert_layer_norm', False)
    args.gelu = getattr(args, 'gelu', True)
    args.apply_bert_init = getattr(args, 'apply_bert_init', True)
