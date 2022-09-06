# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import OrderedDict
import copy
import logging
from typing import Any, Dict, List, Optional
from fairseq.models.transformer.transformer_encoder import TransformerEncoder
import torch
from torch import nn, Tensor

from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer.transformer_decoder import TransformerDecoder

from .s2t_dualinputwavtransformer import DualInputWavTransformerModel, dualinputs2twavtransformer_base_stack
from .s2t_dualinputtransformer import TransformerMultiInputDecoder
from examples.speech_text_joint_to_text.modules.transformer_decoder_layer_context import TransformerDecoderLayerContext


logger = logging.getLogger(__name__)


@register_model("clas_dual_input_wav_transformer")
class CLASDualInputWavTransformerModel(DualInputWavTransformerModel):
    def __init__(self, encoder, decoder, context_encoder):
        super().__init__(encoder, decoder)
        self.nocontext_vector = nn.Parameter(torch.zeros(context_encoder.embed_tokens.embedding_dim))
        self.context_encoder = context_encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        clas_dualinputs2twavtransformer_base_stack(args)

        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        _args = copy.deepcopy(args)
        _args.encoder_layers = args.context_encoder_layers
        context_encoder = TransformerEncoder(_args, decoder.dictionary, decoder.embed_tokens)
        return cls(encoder, decoder, context_encoder)

    @staticmethod
    def add_args(parser):
        DualInputWavTransformerModel.add_args(parser)
        parser.add_argument(
            "--context-activation-fn",
            type=str,
            metavar="FN",
            choices=['softmax', 'entmax15', 'sparsemax'],
            help="type of activation function to use for context",
        )
        parser.add_argument(
            "--add-context-gating",
            action="store_true",
            help="add modality embedding in the ER network",
        )
        parser.add_argument(
            "--context-attention-type",
            type=str,
            metavar="CTX",
            choices=['parallel', 'sequential'],
            help="type of context attention type to use",
        )
        parser.add_argument(
            "--freeze-pretrained-decoder",
            action="store_true",
            help="freeze pretrained decoder params",
        )
        parser.add_argument(
            "--context-encoder-layers",
            type=int,
            metavar="NL",
            help="num of layers for the context encoder",
        )

    @classmethod
    def build_text_decoder(cls, args, tgt_dictionary, dec_emb_share=None):
        dec_emb = (
            nn.Embedding(
                len(tgt_dictionary), args.decoder_embed_dim, tgt_dictionary.pad()
            )
            if dec_emb_share is None
            else dec_emb_share
        )
        return TransformerContextAwareDecoder(args, tgt_dictionary, dec_emb)

    @classmethod
    def build_decoder(cls, args, task):
        decoder = cls.build_text_decoder(args, task.target_dictionary)
        if getattr(args, "load_init_decoder", "") != "":
            pretrained_model_state = checkpoint_utils.load_checkpoint_to_cpu(args.load_init_decoder)
            component_state_dict = OrderedDict()
            for key in pretrained_model_state["model"].keys():
                if key.startswith("decoder.spch_decoder"):
                    # decoder.input_layers.0.0.weight --> decoder.0.0.weight
                    component_subkey = key[21 :]
                    component_state_dict[component_subkey] = pretrained_model_state["model"][key]
                    # Add embeddings for the CONTEXT tag
                    if "embed_tokens." in component_subkey or "output_projection." in component_subkey:
                        component_state_dict[component_subkey] = torch.cat(
                            (component_state_dict[component_subkey],
                             torch.zeros((1, component_state_dict[component_subkey].shape[1]))))

            incompatible_keys = decoder.load_state_dict(component_state_dict, strict=False)
            if len(incompatible_keys.unexpected_keys) != 0:
                logger.error("Cannot load the following keys from checkpoint: {}".format(
                    incompatible_keys.unexpected_keys))
                raise ValueError("Cannot load from checkpoint: {}".format(args.pretrained_model))
            for missing_key in incompatible_keys.missing_keys:
                if 'context' not in missing_key:
                    logger.error("Loaded checkpoint misses the parameter: {}.".format(missing_key))
            if getattr(args, 'freeze_pretrained_decoder', False):
                for p_name, p_val in decoder.named_parameters():
                    if p_name in component_state_dict:
                        p_val.requires_grad = False
        return decoder

    @staticmethod
    def padding_aware_mean(x, padding_mask, batch_dim=1, time_dim=0, channel_dim=2):
        lengths = (1 - padding_mask.long()).sum(dim=-1)
        weight_matrix = torch.zeros(x.shape[batch_dim], x.shape[time_dim])
        for b in range(x.shape[batch_dim]):
            weight_matrix[b, :lengths[b]] = 1.0 / lengths[b].float()
        return x.permute(batch_dim, channel_dim, time_dim).bmm(weight_matrix.to(x.device).unsqueeze(-1)).squeeze(-1)

    def _collate_features(self, features):
        """Convert a list of 2d features into a padded 3d tensor
        Args:
            features (list): list of 2d features of size L[i]*f_dim. Where L[i] is
                length of i-th feature and f_dim is static dimension of features
        Returns:
            3d tensor of size len(features)*len_max*f_dim where len_max is max of L[i]
        """
        len_max = max(f.size(0) for f in features)
        f_dim = features[0].size(1)
        res = features[0].new(len(features), len_max, f_dim).zero_()
        sizes = []
        for i, v in enumerate(features):
            res[i, : v.size(0)] = v
            sizes.append(v.size(0))
        return res, sizes

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        use_encoder_outputs=False,
        src_txt_tokens=None,
        src_txt_lengths=None,
        mode="sup_speech",
        context_list=None,
        context_lengths_list=None,
        **kwargs
    ):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            mode = 'sup_speech' or 'text'

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_txt_tokens=None,
            src_txt_lengths=None,
            **kwargs
        )
        assert len(context_list) == src_tokens.shape[0]
        context_vectors = []
        for context_phrases, context_phrases_lengths in zip(context_list, context_lengths_list):
            if context_phrases is not None:
                ctx_encoder_out = self.context_encoder(
                    context_phrases, context_phrases_lengths, return_all_hiddens=False)
                ctx_phrases_embeddings = self.padding_aware_mean(
                    ctx_encoder_out['encoder_out'][0], ctx_encoder_out['encoder_padding_mask'][0]).detach()
                context_vectors.append(torch.cat((self.nocontext_vector.unsqueeze(0), ctx_phrases_embeddings), dim=0))
            else:
                context_vectors.append(self.nocontext_vector.unsqueeze(0))
        context_embeddings, context_lengths = self._collate_features(context_vectors)  # B x NC x C
        context_embeddings = context_embeddings.transpose(0, 1)
        context_padding_mask = lengths_to_padding_mask(torch.LongTensor(context_lengths).to(context_embeddings.device))
        # has_txt_input = True if src_txt_tokens is not None else False
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            context_embeddings=context_embeddings,
            context_padding_mask=context_padding_mask,
            **kwargs
        )
        if use_encoder_outputs:
            return decoder_out, encoder_out
        return decoder_out


class TransformerMultiInputContextDecoder(TransformerMultiInputDecoder):
    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        has_txt_input=False,
        context_embeddings=None,
        context_padding_mask=None,
        **kwargs
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing. If there are
                two or more input during training, they will share the same prev_output_tokens
            encoder_out (tuple[Tensor]): output from the encoder, used for
                encoder-side attention. It will be tuple if there are more inputs, but a tensor
                if only one input
            incremental_state ([dict]): dictionary used for storing state during
                :ref:`Incremental decoding`. It is only valid for inference, only from single
                input
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`. If there are N inputs, batch will be N bigger than a single input
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        assert not isinstance(encoder_out, EncoderOut)
        if isinstance(encoder_out, tuple):  # training with mulitple input
            rst = []
            assert len(encoder_out) == 2
            for i, eo in enumerate(encoder_out):
                assert incremental_state is None
                if i == 0:
                    rst.append(self.spch_decoder(
                        prev_output_tokens, eo, incremental_state, context_embeddings=context_embeddings, context_padding_mask=context_padding_mask))
                else:
                    rst.append(self.text_decoder(
                        prev_output_tokens, eo, incremental_state, context_embeddings=context_embeddings, context_padding_mask=context_padding_mask))
            dec_out = torch.cat([r[0] for r in rst], dim=0)
            attn_cost = None
            if self.compute_cross_attentive_loss:
                assert isinstance(encoder_out[0], dict)
                if self.cross_attentive_loss_reverse:
                    attn_cost = self.cross_attentive_loss(
                        teacher_states=encoder_out[1]["encoder_states"],  # text_states
                        student_states=encoder_out[0]["encoder_states"],  # spch_states
                        teacher_masking=encoder_out[1]["encoder_padding_mask"],
                        student_masking=encoder_out[0]["encoder_padding_mask"],
                    )
                else:
                    attn_cost = self.cross_attentive_loss(
                        teacher_states=encoder_out[0]["encoder_states"],  # spch_states
                        student_states=encoder_out[1]["encoder_states"],  # text_states
                        teacher_masking=encoder_out[0]["encoder_padding_mask"],
                        student_masking=encoder_out[1]["encoder_padding_mask"],
                    )

            return (dec_out, {"attn_cost": attn_cost})
        else:  # inference or training with one input
            if has_txt_input:
                return self.text_decoder(
                    prev_output_tokens, encoder_out, incremental_state, context_embeddings=context_embeddings, context_padding_mask=context_padding_mask)
            return self.spch_decoder(
                prev_output_tokens, encoder_out, incremental_state, context_embeddings=context_embeddings, context_padding_mask=context_padding_mask)


class TransformerContextAwareDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerContextAwareDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """
    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayerContext(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        context_embeddings: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            context_out (optional): output from the context encoder, used
                for context attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            context_embeddings=context_embeddings,
            context_padding_mask=context_padding_mask,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        context_embeddings: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

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

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        ctx_gates = []
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _, ctx_gate = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                context_embeddings=context_embeddings,
                context_padding_mask=context_padding_mask,
            )
            inner_states.append(x)
            ctx_gates.append(ctx_gate)
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

        return x, {"attn": [attn], "inner_states": inner_states, "ctx_gates": ctx_gates}


@register_model_architecture(
    "clas_dual_input_wav_transformer", "clas_dualinputs2twavtransformer_base_stack"
)
def clas_dualinputs2twavtransformer_base_stack(args):
    args.context_attention_type = getattr(args, "context_attention_type", 'parallel')
    args.add_context_gating = getattr(args, "add_context_gating", False)
    args.context_activation_fn = getattr(args, 'context_activation_fn', 'softmax')
    args.context_encoder_layers = getattr(args, 'context_encoder_layers', 3)
    dualinputs2twavtransformer_base_stack(args)
