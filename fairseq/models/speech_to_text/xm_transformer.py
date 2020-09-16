#!/usr/bin/env python3

import logging
import copy
from typing import Dict, List, Optional, Tuple

from fairseq import utils, checkpoint_utils
from fairseq.models import (FairseqEncoderDecoderModel, FairseqEncoder,
                            register_model, register_model_architecture)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.wav2vec import Wav2VecEncoder, add_common_args
from torch import Tensor
import torch.nn as nn


logger = logging.getLogger(__name__)


class Wav2VecEncoderWithAdaptor(FairseqEncoder):
    def __init__(self, args):
        super().__init__(None)
        self.w2v_encoder = Wav2VecEncoder(args)
        encoder_out_dim = self.w2v_encoder.w2v_model.encoder.embedding_dim
        # Projection + 8x shrinking
        self.adaptor_layers = nn.ModuleList([
            nn.Conv1d(encoder_out_dim, args.decoder_embed_dim * 2, 3,
                      stride=2, padding=1),
            nn.Conv1d(args.decoder_embed_dim, args.decoder_embed_dim * 2, 3,
                      stride=2, padding=1),
            nn.Conv1d(args.decoder_embed_dim, args.decoder_embed_dim * 2, 3,
                      stride=2, padding=1),
        ])
        for k, p in self.w2v_encoder.w2v_model.named_parameters():
            # TODO: make number of layers to finetune an argument
            if not k.startswith('encoder.layers.11'):
                p.requires_grad = False

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        out = self.w2v_encoder.forward(src_tokens, kwargs['padding_mask'],
                                       tbc=True)
        x, enc_padding_mask = out["encoder_out"], out["encoder_padding_mask"]
        # T x B x C -> B x C x T
        x = x.transpose(0, 1).transpose(1, 2).contiguous()
        for adaptor_layer in self.adaptor_layers:
            x = nn.functional.glu(adaptor_layer(x), dim=1)
        # TODO: need to `encoder_padding_mask` if it is not `None`
        # B x C x T -> T x B x C
        x = x.transpose(1, 2).transpose(0, 1).contiguous()

        return EncoderOut(
            encoder_out=x, encoder_padding_mask=enc_padding_mask,
            encoder_embedding=None, encoder_states=None, src_tokens=None,
            src_lengths=None
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            None if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            None if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        return EncoderOut(
            encoder_out=new_encoder_out,
            encoder_padding_mask=new_encoder_padding_mask,
            encoder_embedding=None, encoder_states=None, src_tokens=None,
            src_lengths=None
        )


@register_model("xm_transformer")
class XMTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # wav2vec encoder
        add_common_args(parser)
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        # Transformer decoder
        parser.add_argument("--activation-fn", type=str, default='relu',
                            choices=utils.get_available_activation_fns(),
                            help="activation function to use")
        parser.add_argument("--decoder-dropout", type=float, metavar="D",
                            help="dropout probability")
        parser.add_argument("--decoder-attention-dropout", type=float, metavar="D",
                            help="dropout probability for attention weights")
        parser.add_argument("--decoder-activation-dropout", type=float, metavar="D",
                            help="dropout probability after activation in FFN.")
        parser.add_argument("--decoder-embed-dim", type=int, metavar="N",
                            help="decoder embedding dimension")
        parser.add_argument("--decoder-ffn-embed-dim", type=int, metavar="N",
                            help="decoder embedding dimension for FFN")
        parser.add_argument("--decoder-layers", type=int, metavar="N",
                            help="num decoder layers")
        parser.add_argument("--decoder-attention-heads", type=int, metavar="N",
                            help="num decoder attention heads")
        parser.add_argument("--decoder-normalize-before", action="store_true",
                            help="apply layernorm before each decoder block")
        parser.add_argument("--layernorm-embedding", action="store_true",
                            help="add layernorm to embedding")
        parser.add_argument("--no-scale-embedding", action="store_true",
                            help="if True, dont scale embeddings")
        parser.add_argument("--max-source-positions", default=6000 * 10 * 16,
                            type=int, metavar="N",
                            help="max number of tokens in the source sequence")
        parser.add_argument("--max-target-positions", default=1024, type=int,
                            metavar="N",
                            help="max number of tokens in the target sequence")
        parser.add_argument(
            "--load-pretrained-decoder-from", type=str, metavar="STR",
            help="model to take decoder weights from (for initialization)"
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = Wav2VecEncoderWithAdaptor(args)
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        _args = copy.deepcopy(args)
        _args.dropout = args.decoder_dropout
        _args.attention_dropout = args.decoder_attention_dropout
        _args.activation_dropout = args.decoder_activation_dropout

        decoder = TransformerDecoder(_args, task.target_dictionary,
                                     embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        for p in decoder.parameters():
            p.requires_grad = False
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(task.target_dictionary,
                                               args.decoder_embed_dim)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs,
                                                      sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens,
                                   src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens,
                                   encoder_out=encoder_out)
        return decoder_out


@register_model_architecture(model_name="xm_transformer",
                             arch_name="xm_transformer")
def base_architecture(args):
    # wav2vec encoder
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection",
                                          "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap",
                                           False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = 0.1
    args.layerdrop = getattr(args, "layerdrop", 0.0)

    args.normalize = getattr(args, "normalize", False)

    # mbart decoder
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4*1024)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before',
                                            True)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_attention_dropout = getattr(args, 'decoder_attention_dropout',
                                             0.)
    args.decoder_activation_dropout = getattr(args,
                                              'decoder_activation_dropout', 0.)
    args.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff',
                                           None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', True
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.decoder_output_dim = getattr(args, 'decoder_output_dim',
                                      args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim',
                                     args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
