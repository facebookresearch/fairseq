import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import sys
from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder,
    SinusoidalPositionalEmbedding, LearnedPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, FairseqLanguageModel, register_model,
    register_model_architecture
)

from fairseq.models.transformer import (
    base_lm_architecture #, Embedding, Linear, LayerNorm, PositionalEmbedding
)

#TODO
# Differences from BERT
# -----
# Bias on output tokens
# Separate prediction head layer (dense h*h, ReLU, LayerNorm, project to vocab)
# GeLU activation
# BERT loss is average(lm token loss)+next sentence loss

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
    nn.init.normal_(m.weight, mean=0, std=0.02)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m    

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


@register_model('span_block_autoregressive')
class SpanBlockAutoregressive(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', default=0., type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', default=0., type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-input-output-embed', default=False, action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', default=False, action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--adaptive-input', default=False, action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--ordering', type=str,
                            help='predict words in a random order (left-to-right if false)',
                            default='l2r', choices=['l2r', 'r2l', 'shuffle', 'shifted', 'inside_out', 'multi_inside_out', 'odds_evens', 'l2r_r2l', 'l2r_cloze', 'bi_cloze'])
        parser.add_argument('--universal', action='store_true',
                            help='share parameters across layers')
        parser.add_argument('--stacked-decoder', action='store_true',
                            help='input to each decoder layer is final layer of encoder. Otherwise, each decoder layer conditions on the corresponding encoder layer')
        parser.add_argument('--asymmetric', action='store_true',
                            help='use different parameters for encoder and decoder')
        parser.add_argument('--relative-position', type=str, default='sinusoidal', choices=['none', 'sinusoidal'],
                            help='use relative positions')
        parser.add_argument('--num-segment', type=int, metavar='N', default=3,
                            help='num segment in the input')
        parser.add_argument('--sentence-class-num', type=int, metavar='N', default=0,
                            help='number of classes for sentence task')
        parser.add_argument('--pre-generate-tokens', type=int, metavar='N', default=0,
                            help='Try pre-generating a random number of tokens, that all tokens can condition on')
        parser.add_argument('--span-mask', default=False, action='store_true',
                            help='if use span mask')
        parser.add_argument('--geometric-prob', default = -1.0, type=float, help='1/p is the average span length')
        parser.add_argument('--lower', default=1, type=int, help='lower bound for span length')
        parser.add_argument('--upper', default=10, type=int, help='upper bound for span length') 
        parser.add_argument('--num-saved-mask', type=int)
        parser.add_argument('--fixed-order', default=False, action='store_true', help='fix order for each block')
        parser.add_argument('--noise', default=False, action='store_true', help='if add bert noise to input')
        parser.add_argument('--final-norm', default=False, action='store_true', help='final norm output before softmax')

    def forward(self, src_tokens, segment_labels=None, BlockC=None, **unused):
        return self.decoder(tokens=src_tokens, segment_labels=segment_labels,  BlockC=BlockC, **unused)

    def max_positions(self):
        return self.decoder.max_positions
    def supported_targets(self):
        return {'self', 'future'}
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if hasattr(args, 'no_tie_adaptive_proj') and args.no_tie_adaptive_proj == False:
            # backward compatibility
            args.tie_adaptive_proj = True

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(task.dictionary, eval(args.character_filters),
                                                  args.character_embedding_dim,
                                                  args.embed_dim,
                                                  args.char_embedder_highway_layers,
                                                  )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(len(task.dictionary), task.dictionary.pad(), args.input_dim,
                                         args.adaptive_input_factor, args.embed_dim,
                                         options.eval_str_list(args.adaptive_input_cutoff, type=int))
        else:
            embed_tokens = nn.Embedding(len(task.dictionary), args.embed_dim, task.dictionary.pad())

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.input_dim == args.output_dim

        if not hasattr(args, 'final_norm'):
            args.final_norm = False
        decoder = ShuffleTransformerDecoder(args, task.dictionary, embed_tokens, final_norm=args.final_norm)
        return SpanBlockAutoregressive(decoder)

    @property
    def supported_targets(self):
        return {'self'}


@register_model_architecture('span_block_autoregressive', 'span_block_autoregressive')
def base_lm_architecture(args):
    args.embed_dim = getattr(args, 'embed_dim',768)
    args.ffn_embed_dim = getattr(args, 'ffn_embed_dim', 3072)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.attention_heads = getattr(args, 'attention_heads', 12)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.learned_pos = getattr(args, 'learned_pos', False)
    args.ordering = getattr(args, 'ordering', 'shuffle')
    args.character_embeddings = getattr(args, 'character_embeddings', False)

    args.output_dim = getattr(args, 'output_dim', args.embed_dim)
    args.input_dim = getattr(args, 'input_dim', args.embed_dim)

    # The model training is not stable without this
    args.normalize_before = True

    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)

    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    args.shuffle = getattr(args, 'shuffle', False)
    args.universal = getattr(args, 'universal', False)
    args.stacked_decoder = getattr(args, 'stacked_decoder', False)
    args.asymmetric = getattr(args, 'asymmetric', False)
    args.relative_position = getattr(args, 'relative_position', 'sinusoidal')

    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.pre_generate_tokens = getattr(args, 'pre_generate_tokens', 0)


@register_model_architecture('span_block_autoregressive', 'span_block_autoregressive_big')
def transformer_lm_big(args):
    args.embed_dim = getattr(args, 'embed_dim', 1024)
    args.ffn_embed_dim = getattr(args, 'ffn_embed_dim', 4096)
    args.attention_heads = getattr(args, 'attention_heads', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.decoder_layers = getattr(args, 'decoder_layers', 24)
    base_lm_architecture(args)


class ShuffleTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """


    def __init__(self, args, dictionary, embed_tokens, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.padding_idx = embed_tokens.padding_idx
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.embed_dim
        output_embed_dim = args.output_dim

        padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        #self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.max_positions = args.max_positions

        self.embed_segment = nn.Embedding(
             args.num_segment, embed_dim, self.padding_idx,
        ) if args.num_segment > 0 else None

        self.project_in_dim = nn.Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        #self.prediction_word_embedding = nn.Parameter(torch.Tensor(1, 1, embed_dim).zero_())
        self.prediction_word = dictionary.add_symbol('[unused0]')
        self.embed_positions = PositionalEmbedding(
            self.max_positions, embed_dim, padding_idx,
            left_pad=left_pad,
        ) if not args.no_token_positional_embeddings else None

        def make_layers(args, layers, needs_key_values):
            if args.universal:
                layers = [ShuffleTransformerDecoderLayer(args, needs_key_values=needs_key_values)] * layers
            else:
                layers = [ShuffleTransformerDecoderLayer(args, needs_key_values=needs_key_values) for _ in range(layers)]
            return nn.ModuleList(layers)

        self.stacked_decoder = args.stacked_decoder
        self.encoder_layers = make_layers(args, args.encoder_layers, needs_key_values=True)
        self.decoder_layers = make_layers(args, args.decoder_layers, needs_key_values=False) if args.asymmetric else self.encoder_layers

        if not args.stacked_decoder and args.encoder_layers != args.decoder_layers:
            raise("If not using stacked-decoder, encoder and decoder must have the same number of layers")
        if not args.asymmetric and args.encoder_layers != args.decoder_layers:
            raise("If not using asymmetric, encoder and decoder must have the same number of layers")

        if args.relative_position == 'sinusoidal':
            num_positions = self.max_positions
            sinusoidal_positions = SinusoidalPositionalEmbedding.get_embedding(num_positions, args.embed_dim // args.attention_heads)
            sinusoidal_relative_positions = []
            for i in range(num_positions):
                sinusoidal_relative_positions.append(
                    torch.cat([sinusoidal_positions[num_positions-i:], sinusoidal_positions[:num_positions-i]], 0))
                # Make sentinel token have same relative position to everything
                sinusoidal_relative_positions[-1][0] = 0
                assert sinusoidal_relative_positions[-1].size() == sinusoidal_positions.size()
            sinusoidal_relative_positions = torch.stack(sinusoidal_relative_positions, 0)
            self.sinusoidal_relative_positions = nn.Parameter(sinusoidal_relative_positions)

            assert sinusoidal_relative_positions.size() == (num_positions, num_positions, args.embed_dim // args.attention_heads)
            #assert (sinusoidal_relative_positions[0] == sinusoidal_positions).all()
            assert (sinusoidal_relative_positions[7, 7] == sinusoidal_relative_positions[11, 11]).all()
            assert (sinusoidal_relative_positions[5, 11] == sinusoidal_relative_positions[6, 12]).all()
        else:
            self.sinusoidal_relative_positions = None

        self.adaptive_softmax = None

        self.project_out_dim = nn.Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        self.load_softmax = not getattr(args, 'remove_head', False)
        if self.load_softmax:
            if args.adaptive_softmax_cutoff is not None:
                self.adaptive_softmax = AdaptiveSoftmax(
                   len(dictionary),
                   output_embed_dim,
                   options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                   dropout=args.adaptive_softmax_dropout,
                   adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                   factor=args.adaptive_softmax_factor,
                   tie_proj=args.tie_adaptive_proj,
                 )  
            elif not self.share_input_output_embed:
                self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
                nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)

            #if args.sentence_class_num > 0:
            #    self.sentence_projection_layer = Linear(embed_dim, args.sentence_class_num, bias=False)

        self.normalize = args.normalize_before and final_norm
        if self.normalize:
            self.layer_norm = BertLayerNorm(embed_dim)

        self.apply(self.init_bert_weights)
        self.noise = getattr(args, 'noise', False)
        self.span_mask = getattr(args, 'span_mask', False)
        self.lower = getattr(args, 'lower', None)
        self.upper = getattr(args, 'upper', None)
        self.mask_ratio = getattr(args, 'mask_ratio', 0.6)
        self.lens = list(range(self.lower, self.upper + 1)) if self.lower is not None and self.upper is not None else None
        self.p = getattr(args, 'geometric_prob', -1)
        self.len_distrib = [self.p * (1-self.p)**(i - self.lower) for i in range(self.lower, self.upper + 1)] if self.span_mask else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib] if self.len_distrib is not None else None

        self.save_masks = getattr(args, 'save_masks', True)
        self.fixed_order = getattr(args, 'fixed_order', False)
        if self.save_masks:
            num_saved_mask = getattr(args, 'num_saved_mask', 10000)
            enc_masks = []
            dec_masks = []
            blockCs = []
            if self.fixed_order:
                blockA, blockB, blockC = self.make_block()
                blockA = self.make_ordering(blockA, 'r2l')
                blockB = self.make_ordering(blockB, 'bi_cloze')
                blockC = self.make_ordering(blockC, 'l2r')
                blocks = [blockA, blockB, blockC]
                for idx, given_mask in enumerate(blocks):
                    predict_blocks = [blocks[i] for i in range(3) if i != idx]
                    enc_mask, dec_mask, given_mask = self.make_mask(predict_blocks[0], predict_blocks[1], given_mask)
                    enc_masks.append(enc_mask)
                    dec_masks.append(dec_mask)
                    blockCs.append(given_mask)
            else:
                for i in range (num_saved_mask):
                    blockA, blockB, blockC = self.make_block()
                    blockA = self.make_ordering(blockA, args.ordering)
                    blockB = self.make_ordering(blockB, args.ordering)
                    
                    enc_mask, dec_mask, nonblockC = self.make_mask(blockA, blockB, blockC)
                    enc_masks.append(enc_mask)
                    dec_masks.append(dec_mask)
                    blockCs.append(nonblockC)
            self.register_buffer('enc_masks', torch.stack(enc_masks, 0))
            self.register_buffer('dec_masks', torch.stack(dec_masks, 0))
            self.register_buffer('blockC', torch.stack(blockCs, 0))

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, tokens, segment_labels, BlockC=None, apply_mask=True, target=None, mask=None, **unused):

        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        #assert (BlockC is None and not apply_mask) or (BlockC is not None and apply_mask)
        #assert max(BlockC) < tokens.size(1) 
        bsz, num_words = tokens.size()
        #tokens = tokens[:, :self.max_positions]

        # get padding mask
        padding_mask = tokens.eq(self.padding_idx)
        if segment_labels is not None:
            segment_labels = segment_labels[:, : self.max_positions]
        assert not torch.isnan(tokens).any()
        # embed tokens
        assert int(tokens.max()) < self.embed_tokens.weight.size(0), 'max is {} and size is {}'.format(tokens.max(),self.embed_tokens.weight.size(0))
        #x_enc = self.embed_tokens(tokens)
        #print (self.embed_tokens.weight.data.norm())
        assert not torch.isnan(self.embed_tokens.weight.data).any()
        x_enc = self.embed_tokens(tokens)
        assert not torch.isnan(x_enc).any()
        #x_enc *= self.embed_scale
        assert not torch.isnan(x_enc).any()
        # psotion embed
        positions = self.embed_positions(tokens) if self.embed_positions is not None else None
        # segment embed
        segment_embeddings = (
            self.embed_segment(segment_labels.long())
            if self.embed_segment is not None
            else None
        )

        nonBlockC = None
        if not apply_mask or not self.save_masks:
            if mask is None:
                enc_mask = x_enc.new(num_words, num_words).fill_(0).unsqueeze(0)
                dec_mask = x_enc.new(num_words, num_words).fill_(0).unsqueeze(0)
            else:
                enc_mask, dec_mask = mask
                enc_mask = enc_mask.type_as(x_enc)
                dec_mask = dec_mask.type_as(x_enc)
        else:
            nonBlockC, enc_mask, dec_mask = self.genetate_mask(x_enc)
   
        if apply_mask or (mask is not None):
            mask_tokens = tokens.new(bsz, num_words).fill_(self.prediction_word)
            
            if self.noise:
                noise_num = int(sum(nonBlockC).item() * 0.1)
                perm = torch.randperm(len(nonBlockC))
                predict_tokens = nonBlockC.nonzeros()
                perm = torch.randperm(sum(nonBlockC).item())
                predict_tokens = predict_tokens[perm]
                random_tokens = predict_tokens[noise_num:noise_num*2]
                unchange_tokens = predict_tokens[:noise_num]
                mask_tokens[unchange_tokens] = tokens[unchange_tokens].clone()
                mask_tokens[random_tokens] = tokens.new(bsz, noise_num).random_(0, len(self.dictionary))
                  
            x_dec = self.embed_tokens(mask_tokens)
        else:
            x_dec = x_enc.clone()

        #assert self._test_mask()
        assert enc_mask.size(0) == bsz or enc_mask.size(0) == 1
        assert enc_mask.size(1) == num_words
        assert enc_mask.size(2) == num_words

        if positions is not None:
            x_enc = x_enc + positions
            x_dec = x_dec + positions
        assert not torch.isnan(x_enc).any()
        if segment_embeddings is not None:
            x_enc = x_enc + segment_embeddings
            x_dec = x_dec + segment_embeddings
        if self.project_in_dim is not None:
            x_enc = self.project_in_dim(x_enc)
        assert not torch.isnan(x_enc).any()

        if nonBlockC is not None:
            x_dec = x_dec[:, nonBlockC]
            if target is not None:
                target = target[:, nonBlockC]

        x_enc = F.dropout(x_enc, p=self.dropout, training=self.training)
        x_dec = F.dropout(x_dec, p=self.dropout, training=self.training)
       
        # B x T x C -> T x B x C
        x_dec = x_dec.transpose(0, 1)
        x_enc = x_enc.transpose(0, 1)

        assert not torch.isnan(x_enc).any()
        assert not torch.isnan(x_dec).any()
         
        attn = None

        """if not apply_mask:
            enc_mask = enc_mask * 0
            dec_mask = dec_mask * 0"""
        if self.sinusoidal_relative_positions is not None:
            sinusoidal_relative_positions = self.sinusoidal_relative_positions[:num_words,:num_words]#.float()
        else:
            sinusoidal_relative_positions = None
        if self.stacked_decoder:
            # encoder layers
            for i, layer in enumerate(self.encoder_layers):
                queries_enc, keys, values = layer.self_attn.in_proj_qkv(
                    layer.maybe_layer_norm(layer.self_attn_layer_norm, x_enc, before=True))
                x_enc, _ = layer(
                    keys, values, queries_enc, x_enc, sinusoidal_relative_positions,
                    self_attn_mask=enc_mask, self_attn_padding_mask= padding_mask,
                )
                assert not torch.isnan(x_enc).any()

            # decoder layers
            # just use key/values from final encode layer
            keys, values = layer.self_attn.in_proj_kv(
                layer.maybe_layer_norm(layer.self_attn_layer_norm, x_enc, before=True))

            for i, dec_layer in enumerate(self.decoder_layers):
                queries_dec = dec_layer.self_attn.in_proj_q(
                    dec_layer.maybe_layer_norm(dec_layer.self_attn_layer_norm, x_dec, before=True))
                x_dec, _ = dec_layer(
                    keys, values, queries_dec, x_dec, sinusoidal_relative_positions,
                    self_attn_mask=dec_mask, self_attn_padding_mask= padding_mask,
                )

                assert not torch.isnan(x_dec).any()
        else:
            # encoder layers
            layer = self.encoder_layers[0]
            queries_enc, keys, values = layer.self_attn.in_proj_qkv(layer.maybe_layer_norm(layer.self_attn_layer_norm, x_enc, before=True))
            
            for i, layer in enumerate(self.encoder_layers):

                x_enc, _ = layer(
                    keys, values, queries_enc, x_enc, sinusoidal_relative_positions,
                    self_attn_mask=enc_mask,  self_attn_padding_mask= padding_mask,
                )
                assert not torch.isnan(x_enc).any()

                # keys and values are the same for encoder layer i+1 and decoder layer i
                x_enc_normed = layer.maybe_layer_norm(layer.self_attn_layer_norm, x_enc, before=True)
                if i == len(self.encoder_layers) - 1:
                    # At final layer, we don't need an encoder query
                    keys, values = layer.self_attn.in_proj_kv(x_enc_normed)
                else:
                    queries_enc, keys, values = layer.self_attn.in_proj_qkv(x_enc_normed)

                dec_layer = self.decoder_layers[i]
                queries_dec = dec_layer.self_attn.in_proj_q(dec_layer.maybe_layer_norm(dec_layer.self_attn_layer_norm, x_dec, before=True))
                x_dec, _ = dec_layer(
                        keys, values, queries_dec, x_dec, sinusoidal_relative_positions[nonBlockC],
                    self_attn_mask=dec_mask,  self_attn_padding_mask=padding_mask,
                )
                assert not torch.isnan(x_dec).any()
        if self.normalize:
            x_dec = self.layer_norm(x_dec)

        # T x B x C -> B x T x C
        x_dec = x_dec.transpose(0, 1)
        if self.project_out_dim is not None:
            x_dec = self.project_out_dim(x_dec)

        #sentence_rep = x_dec[:, 1, :]
        if self.load_softmax:
            if self.adaptive_softmax is None:
                # project back to size of vocabulary
                if self.share_input_output_embed:
                    x_dec = F.linear(x_dec, self.embed_tokens.weight)
                else:
                    x_dec = F.linear(x_dec, self.embed_out)

        assert not torch.isnan(x_dec).any()

        return x_dec, {'attn': attn, 'target':target} #, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_positions - 1
        return min(self.max_positions, self.embed_positions.max_positions()) - 1

    def make_block(self):
        if not self.span_mask:
            blockA_size = (self.max_positions) // 3
            blockB_size = self.max_positions - blockA_size
            if self.fixed_order:
                order = torch.range(0, self.max_positions).long()
            else:
                order = torch.randperm(self.max_positions)
            blockA = order[: blockA_size]
            blockB = order[blockA_size:blockB_size]
            blockC = order[blockB_size:]
        else:
            mask_num = math.ceil((self.max_positions) * self.mask_ratio)
            mask = set()
            masks_span = []
            while len(mask) < mask_num:
                current_span = []
                pointer = np.random.choice(self.max_positions)
                if pointer in mask:
                    continue
                num_words = 0
                span_len = np.random.choice(self.lens, p=self.len_distrib)
                while num_words < span_len and pointer < self.max_positions and len(mask) < mask_num:
                    if pointer not in mask:
                        mask.add(pointer)
                        current_span.append(pointer)
                    else:
                        break
                    num_words += 1
                    pointer += 1
                masks_span.append(current_span)
            blockC = torch.LongTensor([i for i in range(self.max_positions) if i not in mask])
            assert sum(len(j) for j in masks_span) == len(mask)
            total_span = len(masks_span)
            blockA = masks_span[:len(masks_span) // 2]
            blockB = masks_span[len(masks_span) // 2:]
            blockA = torch.LongTensor([i for j in blockA for i in j])
            blockB = torch.LongTensor([i for j in blockB for i in j])

        return blockA, blockB, blockC

    def genetate_mask(self, tensor):
        bsz, dim, _ = tensor.size()
        idx = random.randint(0, len(self.enc_masks) - 1)
        nonblockC = self.blockC[idx][:dim]
        while sum(nonblockC).item() == 0:
            idx = random.randint(0, len(self.enc_masks) - 1)
            nonblockC = self.blockC[idx][:dim]
        enc_mask_byte = self.enc_masks[idx][:dim, :dim]
        dec_mask_byte = self.dec_masks[idx][:dim, :dim]
        enc_mask = utils.fill_with_neg_inf(tensor.new(dim, dim))
        dec_mask = utils.fill_with_neg_inf(tensor.new(dim, dim))
        enc_mask[~enc_mask_byte] = 0
        dec_mask[~dec_mask_byte] = 0
        return nonblockC.cuda(), enc_mask.unsqueeze(0), dec_mask[nonblockC].unsqueeze(0)

    def make_mask(self, blockA, blockB, blockC):
        blockA_size = len(blockA)
        blockB_size = len(blockB)
        blockC_size = len(blockC)
        dim = blockA_size + blockB_size + blockC_size
        mask = torch.empty(dim, dim).fill_(1).byte()
        # Construct encoder mask
        enc_mask = mask.clone()
        # Block C can see all the positions in C
        #self.mask_helper(enc_mask, BlockC) 
        for idx in blockC.data:
            enc_mask[idx, blockC] = 0
        # Block B can see block C and positions up to the position(i) in B
        block_B = torch.cat([blockC, blockB])
        for idx_idx, idx in enumerate(blockB.data):
            enc_mask[idx, block_B[:blockC_size+idx_idx+1]] = 0
        # Block A can see block C and positions up to the position(i) in A
        block_A = torch.cat([blockC, blockA])
        for idx_idx, idx in enumerate(blockA.data):
            enc_mask[idx, block_A[:blockC_size+idx_idx+1]] = 0
        
        assert (torch.diag(enc_mask)==0).all()

        # Construct decoder mask
        # Block B can see block C and A and positions up to the position(i-1) in B
        dec_mask = mask.clone()
        for idx in blockC:
            dec_mask[idx, blockC] = 0
        block_B = torch.cat([blockC, blockA, blockB])
        for idx_idx, idx in enumerate(blockB):
            dec_mask[idx, block_B[:blockC_size+blockA_size+idx_idx]] = 0
        # Block B can see block C and B and positions up to the position(i-1) in A
        block_A = torch.cat([blockC, blockB, blockA])
        for idx_idx, idx in enumerate(blockA):
            dec_mask[idx, block_A[:blockC_size+blockB_size+idx_idx]] = 0

        #assert (torch.diag(dec_mask) != 0).all()
        nonBlockC = torch.zeros(self.max_positions, dtype=torch.uint8)
        nonBlockC[blockA] = 1
        nonBlockC[blockB] = 1
        return enc_mask, dec_mask, nonBlockC

    def _test_mask(self):
        enc, dec = self.make_mask([4,2],[1,5],[0,3], torch.tensor([1,2,3,4,5,6]))
        enc, dec = enc < 0, dec < 0
        enc_ans = torch.tensor([[[0, 1, 1, 0, 1, 1],
                              [0, 0, 1, 0, 1, 1],
                              [0, 1, 0, 0, 0, 1],
                              [0, 1, 1, 0, 1, 1],
                              [0, 1, 1, 0, 0, 1],
                              [0, 0, 1, 0, 1, 0]]], dtype=torch.uint8).type_as(enc)


        assert torch.equal(enc, enc_ans)

        assert torch.equal(dec,
                torch.tensor([[[0, 1, 1, 0, 1, 1],
                              [0, 1, 0, 0, 0, 1],
                              [0, 0, 1, 0, 0, 0],
                              [0, 1, 1, 0, 1, 1],
                              [0, 0, 1, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]]], dtype=torch.uint8))
        return True

    def make_ordering(self, idxs, order_type):
        
        def l2r(idxs):
            return idxs.new(sorted(idxs.tolist()))

        def cloze(idxs):
            standard = []
            cloze = []
            for i in idxs:
                if random.random() < 0.15:
                    cloze.append(i)
                else:
                    standard.append(i)
            return idxs.new(standard + cloze)

        def l2r_or_r2l(idxs):
            order = l2r(idxs)
            if random.random() < 0.5:
                order = list(reversed(order.tolist()))
            return idxs.new(order)
        
        if order_type == 'l2r':
            order = l2r(idxs)
        elif order_type == 'r2l':
            order = reversed(l2r(idxs))
        elif order_type == 'l2r_r2l':
            order = l2r_or_r2l(idxs)
        elif order_type == 'l2r_cloze':
            order = cloze(l2r(idxs))
        elif order_type == 'bi_cloze':
            order = cloze(l2r_or_r2l(idxs))
        elif order_type == 'shuffle':
            order = idxs 
        elif order_type == 'shifted':
            xs = l2r(idxs)
            split = random.randint(0, len(xs))
            order = xs[split:] + xs[:split]
        else:
            raise("Unexpected order: " + order_type)

        return order

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        pass


class ShuffleTransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args, needs_key_values=True):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.attention_heads,
            dropout=args.attention_dropout, needs_key_values=needs_key_values
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.normalize_before

        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, args.ffn_embed_dim)
        self.fc2 = nn.Linear(args.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, keys, values, queries, x, relative_position_keys,
                prev_self_attn_state=None, prev_attn_state=None, self_attn_mask=None,
                self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        assert not torch.isnan(x).any()
        x, attn = self.self_attn(
            query=queries,
            key=keys,
            value=values,
            relative_position_keys=relative_position_keys,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,

        )
        
        assert not torch.isnan(x).any()
        x = F.dropout(x, p=self.dropout, training=self.training)#, inplace=True)
        x = x + residual
        assert not torch.isnan(x).any()
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        assert not torch.isnan(x).any()
        x = gelu(self.fc1(x))#, inplace=True)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)#, inplace=True)
        x = self.fc2(x)
        assert not torch.isnan(x).any()
        x = F.dropout(x, p=self.dropout, training=self.training)#, inplace=True)
        x = residual + x
        assert not torch.isnan(x).any()
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        assert not torch.isnan(x).any()
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, needs_key_values=True):#, fp16=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5


        #self.in_proj_q = nn.Linear(embed_dim, embed_dim, bias=bias)

        num_functions = 3 if needs_key_values else 1
        self.in_proj_weight = Parameter(torch.Tensor(num_functions * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(num_functions * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        self.in_proj_weight.data.normal_(mean=0.0, std=0.02)
        self.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, relative_position_keys,
                key_padding_mask=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        src_len, bsz, embed_dim = key.size()
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()
        assert not torch.isnan(key).any()
        assert not torch.isnan(value).any()
        assert not torch.isnan(query).any()
        query *= self.scaling
        assert not torch.isnan(query).any()

        if self.bias_k is not None:
            assert self.bias_v is not None
            key = torch.cat([key, self.bias_k.repeat(1, bsz, 1)])
            value = torch.cat([value, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        query = query.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        assert not torch.isnan(query).any()
        if key is not None:
            key = key.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if value is not None:
            value = value.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if attn_mask is not None:
            if len(attn_mask.size()) == 3 and attn_mask.size(0) != 1:
                attn_mask = attn_mask.repeat(1, self.num_heads, 1).view(bsz * self.num_heads, tgt_len, src_len)
            attn_weights = torch.baddbmm(attn_mask, query, key.transpose(1, 2))
        else:
            attn_weights = torch.bmm(query, key.transpose(1, 2))
        assert not torch.isnan(attn_weights).any()

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]


        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            #attn_weights = attn_weights.float().masked_fill(
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if relative_position_keys is not None:
            # batch * from * dim, from * to * dim --> batch * from * to
            relative_position_weights = torch.einsum('bfd,ftd->bft', [query.float(), relative_position_keys.float()]).type_as(attn_weights)
            assert relative_position_weights.size() == (bsz * self.num_heads, tgt_len, src_len)
            #attn_weights = attn_weights.float()
            attn_weights += relative_position_weights
            #attn_weights = attn_weights.type_as(query)
        
        attn_weights = torch.clamp(attn_weights, -10000., 10000.)
        attn_weights = F.softmax(attn_weights, dim=-1)#.type_as(query)
        #attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(query)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)#, inplace=self.fp16)
        assert not torch.isnan(attn_weights).any()
        attn = torch.bmm(attn_weights, value)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        if relative_position_keys is not None:
            relative_position_vals = torch.einsum('bft,ftd->bfd', [attn_weights.float(),
                                                                   relative_position_keys.float()]).type_as(attn_weights)
            attn = attn + relative_position_vals

        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


