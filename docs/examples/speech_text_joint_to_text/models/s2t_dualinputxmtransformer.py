# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
)
from fairseq.models.speech_to_text import Wav2VecEncoderWithAdaptor
from fairseq.models.speech_to_text.xm_transformer import (
    set_default_adaptor_args,
    set_default_w2v_encoder_args,
    need_finetuning
)
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.models.wav2vec import TransformerSentenceEncoderLayer
from fairseq.utils import safe_hasattr

from .s2t_dualinputtransformer import (
    DualInputS2TTransformerModel,
    TransformerMultiInputDecoder,
    DualInputEncoder,
)


class TransformerSentenceEncoderLayerStd(TransformerSentenceEncoderLayer):
    def __init__(self, sent_enc_layer):
        super(TransformerSentenceEncoderLayer, self).__init__()
        self.embedding_dim = sent_enc_layer.embedding_dim
        self.dropout = sent_enc_layer.dropout
        self.activation_dropout = sent_enc_layer.activation_dropout

        # Initialize blocks
        self.activation_fn = sent_enc_layer.activation_fn
        self.self_attn = sent_enc_layer.self_attn

        self.dropout1 = sent_enc_layer.dropout1
        self.dropout2 = sent_enc_layer.dropout2
        self.dropout3 = sent_enc_layer.dropout3

        self.layer_norm_first = sent_enc_layer.layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = sent_enc_layer.self_attn_layer_norm
        self.fc1 = sent_enc_layer.fc1
        self.fc2 = sent_enc_layer.fc2

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = sent_enc_layer.final_layer_norm

    def forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_weights=None,
        att_args=None,
    ):
        x, attn = super().forward(
            x, self_attn_mask, self_attn_padding_mask, need_weights, att_args
        )
        return x


# TODO retire SharedEncoder
class SharedEncoder(FairseqEncoder):
    def __init__(self, wav2vec_enc, mbart_enc, adaptor, shared_layers):
        super().__init__(None)
        self.w2v_encoder = wav2vec_enc
        self.shared_layers = self.w2v_encoder.w2v_model.encoder.layers[-shared_layers:]
        self.w2v_encoder.w2v_model.encoder.layers = (
            self.w2v_encoder.w2v_model.encoder.layers[:-shared_layers]
        )
        self.adaptor = adaptor
        if self.shared_layers[-1].layer_norm_first:
            self.final_layer_norm = mbart_enc.layer_norm
        else:
            mbart_enc.layer_norm = None
            self.final_layer_norm = None
        shared_layer_from = len(mbart_enc.layers) - shared_layers
        if shared_layer_from < 0:
            shared_layer_from = 0
        for layer_id, layer in enumerate(self.shared_layers):
            mbart_enc.layers[
                shared_layer_from + layer_id
            ] = TransformerSentenceEncoderLayerStd(layer)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        padding_mask = lengths_to_padding_mask(src_lengths)
        if not padding_mask.any():
            padding_mask = None

        out = self.w2v_encoder.forward(src_tokens, padding_mask, tbc=True)
        x = out["encoder_out"]
        enc_padding_mask = None
        if out["encoder_padding_mask"] is not None:
            enc_padding_mask = out["encoder_padding_mask"].transpose(
                0, 1
            )  # T X B --> B X T

        x, enc_padding_mask = self.adaptor(x, enc_padding_mask)
        for layer in self.shared_layers:
            x, _ = layer(x, enc_padding_mask)
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [enc_padding_mask]
            if enc_padding_mask is not None
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


class StackedWav2VecEncoderWithAdaptor(FairseqEncoder):
    def __init__(
        self,
        wav2vec_enc,
        mbart_enc_layers,
        mbart_layer_norm,
        adaptor,
        drop_w2v_layers=0,
    ):
        super().__init__(None)
        self.w2v_encoder = wav2vec_enc
        self.adaptor = adaptor
        self.mbart_encoder_layers = mbart_enc_layers
        self.final_layer_norm = mbart_layer_norm
        if drop_w2v_layers > 0:
            self.w2v_encoder.w2v_model.encoder.layers = (
                self.w2v_encoder.w2v_model.encoder.layers[:-drop_w2v_layers]
            )

    def forward(self, src_tokens, src_lengths=None, return_all_hiddens=False, **kwargs):
        padding_mask = lengths_to_padding_mask(src_lengths)
        if not padding_mask.any():
            padding_mask = None

        out = self.w2v_encoder.forward(src_tokens, padding_mask, tbc=True)
        x = out["encoder_out"]
        enc_padding_mask = None
        if out["padding_mask"] is not None:
            enc_padding_mask = out["padding_mask"]  # B X T

        x, enc_padding_mask = self.adaptor(x, enc_padding_mask)
        encoder_states = []
        for layer in self.mbart_encoder_layers:
            x = layer(x, enc_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [enc_padding_mask]
            if enc_padding_mask is not None
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


# Note:
# dual input transformer:
#    encoder: wav2vec for speech + mbart encoder for text
#    decoder: mbart decoder  for text
@register_model("dual_input_xm_transformer")
class DualInputXMTransformerModel(DualInputS2TTransformerModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # wav2vec encoder
        Wav2VecEncoderWithAdaptor.add_args(parser)
        # add_decoder_args(parser)
        # mbart Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )

        parser.add_argument(
            "--mbart-dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--mbart-attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--mbart-activation-dropout",
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
            "--load-pretrained-mbart-from",
            type=str,
            metavar="STR",
            help="model to take text encoder decoder weights from (for initialization)",
        )
        # parser.add_argument("--finetune-w2v-params", type=str, metavar="STR",
        #                    help="comma-separated param strings to finetune.")
        parser.add_argument(
            "--finetune-mbart-decoder-params",
            type=str,
            metavar="STR",
            help="comma-separated param strings to finetune.",
        )
        parser.add_argument(
            "--finetune-mbart-encoder-params",
            type=str,
            metavar="STR",
            help="comma-separated param strings to finetune.",
        )
        parser.add_argument(
            "--skip-encoder-projection",
            action="store_true",
            help="skip the projection layer in encoder",
        )

        parser.add_argument(
            "--enc-grad-mult",
            type=float,
            metavar="V",
            default=1.0,
            help="multiply enc1 and enc2 gradient by V",
        )
        parser.add_argument(
            "--enc2-along-grad-mult",
            type=float,
            metavar="V",
            default=1.0,
            help="multiply enc2 gradient by V if only enc2 is used",
        )
        parser.add_argument(
            "--text-input-cost-ratio",
            type=float,
            default=1.0,
            metavar="V",
            help="text input cost ratio relative to speech input cost",
        )
        parser.add_argument(
            "--stack-w2v-mbart-encoder",
            action="store_true",
            help="stack w2v and mbart encoder",
        )
        parser.add_argument(
            "--stack-w2v-mbart-nonorm-encoder",
            action="store_true",
            help="stack w2v and mbart encoder",
        )
        parser.add_argument(
            "--no-final-norm-decoder", action="store_true", help="no layer norm"
        )
        parser.add_argument(
            "--drop-w2v-layers",
            type=int,
            default=0,
            metavar="N",
            help="drop w2v encoder layers",
        )

        parser.add_argument(
            "--share-w2v-text-encoder",
            action="store_true",
            help="share w2v encoder layers with text encoder",
        )
        parser.add_argument(
            "--shared-w2v-layers",
            type=int,
            default=0,
            metavar="N",
            help="shared encoder layers from w2v encoder",
        )

    @classmethod
    def build_encoder(cls, args, task):
        _args = copy.deepcopy(args)
        _args.dropout = args.mbart_dropout
        _args.attention_dropout = args.mbart_attention_dropout
        _args.activation_dropout = args.mbart_activation_dropout
        _args.max_source_positions = 1024
        enc_emb = nn.Embedding(
            len(task.src_dict), _args.encoder_embed_dim, task.src_dict.pad()
        )
        text_encoder = TransformerEncoder(_args, task.src_dict, enc_emb)
        spch_encoder = Wav2VecEncoderWithAdaptor(args)
        if getattr(args, "load_pretrained_mbart_from", None):
            text_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=text_encoder, checkpoint=args.load_pretrained_mbart_from
            )
        if getattr(args, "stack_w2v_mbart_encoder", False):
            assert getattr(args, "share_w2v_text_encoder", False) is False
            spch_encoder = StackedWav2VecEncoderWithAdaptor(
                spch_encoder.w2v_encoder,
                text_encoder.layers,
                text_encoder.layer_norm,
                spch_encoder.adaptor,
                args.drop_w2v_layers,
            )
        elif getattr(args, "stack_w2v_mbart_nonorm_encoder", False):
            text_encoder.layer_norm = None
            spch_encoder = StackedWav2VecEncoderWithAdaptor(
                spch_encoder.w2v_encoder,
                text_encoder.layers,
                text_encoder.layer_norm,
                spch_encoder.adaptor,
                args.drop_w2v_layers,
            )
        elif getattr(args, "share_w2v_text_encoder", False):
            spch_encoder = SharedEncoder(
                spch_encoder.w2v_encoder,
                text_encoder,
                spch_encoder.adaptor,
                args.shared_w2v_layers,
            )

        for k, p in spch_encoder.named_parameters():
            # Freeze pretrained models by default
            if safe_hasattr(
                args, "finetune_w2v_params"
            ) and need_finetuning(args.finetune_w2v_params, k):
                p.requires_grad = True
            else:
                p.requires_grad = False
        for k, p in text_encoder.named_parameters():
            # Freeze pretrained models by default
            if safe_hasattr(
                args, "finetune_mbart_encoder_params"
            ) and need_finetuning(
                args.finetune_mbart_encoder_params, k
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False
        cross_attentive_loss_before_last_layer = (
            0 if getattr(args, "attentive_cost_regularization", 0.0) > 0.0 else -1
        )
        encoder = DualInputEncoder(
            args,
            spch_encoder,
            text_encoder,
            task.src_dict,
            cross_attentive_loss_before_last_layer,
        )
        return encoder

    @classmethod
    def build_decoder(cls, args, task):
        _args = copy.deepcopy(args)
        _args.dropout = args.mbart_dropout
        _args.attention_dropout = args.mbart_attention_dropout
        _args.activation_dropout = args.mbart_activation_dropout
        _args.max_target_positions = 1024
        dec_emb = nn.Embedding(
            len(task.tgt_dict), _args.encoder_embed_dim, task.tgt_dict.pad()
        )
        decoder = TransformerDecoder(_args, task.tgt_dict, dec_emb)
        if getattr(args, "load_pretrained_mbart_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_mbart_from
            )
        if getattr(args, "no_final_norm_decoder", False):
            decoder.layer_norm = None
        for k, p in decoder.named_parameters():
            # Freeze pretrained models by default
            if safe_hasattr(
                args, "finetune_mbart_decoder_params"
            ) and need_finetuning(
                args.finetune_mbart_decoder_params, k
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False

        compute_cross_attentive_loss = (
            True if getattr(args, "attentive_cost_regularization", 0.0) > 0.0 else False
        )
        cross_attentive_loss_without_norm = getattr(
            args, "attentive_cost_without_normalize", False
        )
        cross_attentive_loss_reverse = (
            False  # getattr(args, "attentive_cost_reverse", False)
        )
        decoder = TransformerMultiInputDecoder(
            dictionary=task.target_dictionary,
            spch_decoder=decoder,
            text_decoder=decoder,
            compute_cross_attentive_loss=compute_cross_attentive_loss,
            cross_attentive_loss_with_norm=True
            if not cross_attentive_loss_without_norm
            else False,
            cross_attentive_loss_reverse=cross_attentive_loss_reverse,
        )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        dualinputxmtransformer_base(args)

        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(encoder, decoder)


@register_model_architecture("dual_input_xm_transformer", "dualinputxmtransformer_base")
def dualinputxmtransformer_base(args):
    # wav2vec encoder
    set_default_w2v_encoder_args(args)
    set_default_adaptor_args(args)

    # mbart model
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(
        args, "encoder_ffn_embed_dim", 4 * args.encoder_embed_dim
    )
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.adaptive_input = getattr(args, "adaptive_input", False)

    args.mbart_attention_dropout = getattr(args, "mbart_attention_dropout", 0.0)
    args.mbart_activation_dropout = getattr(args, "mbart_activation_dropout", 0.0)
    args.mbart_dropout = getattr(args, "mbart_dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
