# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import namedtuple

import torch
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.speech_to_text import (
    TransformerDecoder,
    S2TTransformerEncoder,
)
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import (
    TransformerEncoderLayer,
    GradMultiply,
    LayerNorm,
)

logger = logging.getLogger(__name__)


class SpeechEoSEncoder(FairseqEncoder):
    def __init__(self, encoder, eos_num, feat_dim, adapter_type="None", adapter_dim=0):
        super().__init__(None)
        self.encoder = encoder
        self.eos_num = eos_num  # downsampling rate for speech input feature
        self.eos_emb = (
            nn.Parameter(torch.zeros(1, feat_dim), requires_grad=True)
            if eos_num > 0
            else None
        )
        self.adapter = self.add_adapter(adapter_type, adapter_dim)

    def add_adapter(self, adapter_type, adapter_dim):
        def _make_identity(linear, eps=1e-5):
            assert isinstance(linear, nn.Linear)
            linear.weight.data.mul_(eps)
            linear.weight.data.fill_diagonal_(1.0)
            if linear.bias is not None:
                linear.bias.data.mul_(eps)

        adapter = None
        if adapter_type == "Linear":
            assert adapter_dim > 0
            adapter = nn.Sequential(
                nn.Linear(adapter_dim, adapter_dim), LayerNorm(adapter_dim)
            )
            # initialize the adapter as identity matrix first
            _make_identity(adapter[0])

        elif adapter_type == "MLP":
            assert adapter_dim > 0
            # assume the model is pre-norm model
            adapter = nn.Sequential(
                nn.Linear(adapter_dim, 2 * adapter_dim),
                nn.ReLU(),
                nn.Linear(2 * adapter_dim, adapter_dim),
                LayerNorm(adapter_dim),
            )
            _make_identity(adapter[0])
            _make_identity(adapter[2])
        return adapter

    def add_eos(self, src_tokens, src_lengths):
        bsz, max_seq_len, fdim = src_tokens.size()
        if self.eos_num > 0:
            src_token_eos = torch.zeros(
                [bsz, max_seq_len + self.eos_num, fdim],
                dtype=src_tokens.dtype,
                device=src_tokens.device,
            )
            src_token_eos[:, :max_seq_len] = src_tokens
            for bi in range(bsz):
                src_token_eos[bi][
                    src_lengths[bi] : src_lengths[bi] + self.eos_num
                ] = self.eos_emb.expand(self.eos_num, fdim)
            src_lengths = src_lengths + self.eos_num
            src_tokens = src_token_eos
        return src_tokens, src_lengths

    def apply_adapter(self, enc_out):
        if self.adapter is None:
            return enc_out
        rst = self.adapter(enc_out.encoder_out)
        if enc_out.encoder_padding_mask is not None:
            rst.masked_fill_(
                enc_out.encoder_padding_mask.transpose(0, 1).unsqueeze(-1), 0
            )
        return EncoderOut(
            encoder_out=rst,
            encoder_padding_mask=enc_out.encoder_padding_mask,
            encoder_embedding=enc_out.encoder_embedding,
            encoder_states=enc_out.encoder_states,
            src_tokens=enc_out.src_tokens,
            src_lengths=enc_out.src_lengths,
        )

    def forward(self, src_tokens, src_lengths=None, return_all_hiddens=False, **kwargs):
        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """
        src_tokens, src_lengths = self.add_eos(src_tokens, src_lengths)
        enc_out = self.encoder(src_tokens, src_lengths, return_all_hiddens)
        enc_out = self.apply_adapter(enc_out)
        return enc_out

    def reorder_encoder_out(self, encoder_out, new_order):
        return self.encoder.reorder_encoder_out(encoder_out, new_order)


class DualInputEncoder(FairseqEncoder):
    def __init__(
        self,
        args,
        spch_encoder,
        text_encoder,
        dictionary,
        cross_attentive_loss_before_last_layer=-1,
    ):
        super().__init__(dictionary)

        self.spch_encoder = spch_encoder
        self.text_encoder = text_encoder
        self.enc_grad_mult = args.enc_grad_mult
        self.cross_attentive_loss_before_last_layer = (
            cross_attentive_loss_before_last_layer
        )
        self.use_cross_attentive_loss = (
            False if cross_attentive_loss_before_last_layer <= -1 else True
        )
        self.enc2_along_grad_mult = args.enc2_along_grad_mult

    @classmethod
    def set_shared_layer(cls, share_level, src_layer, tgt_layer):
        """
        share parameters from tgt_layer to src_layer
        share_level:
            0: share everything
            1: share everything but different model
            2: share weight but not bias, layernorm
        """
        if share_level == 0:
            return tgt_layer
        if isinstance(src_layer, nn.Linear):
            return tgt_layer
        if isinstance(src_layer, TransformerEncoderLayer):
            assert src_layer.embed_dim == tgt_layer.embed_dim
            assert src_layer.normalize_before == tgt_layer.normalize_before
            if share_level == 1:
                src_layer.fc1 = tgt_layer.fc1
                src_layer.fc2 = tgt_layer.fc2
                src_layer.self_attn = tgt_layer.self_attn
                src_layer.final_layer_norm = tgt_layer.final_layer_norm
                src_layer.self_attn_layer_norm = tgt_layer.self_attn_layer_norm
                src_layer.layernorm_embedding = tgt_layer.layernorm_embedding
            else:
                src_layer.fc1.weight = tgt_layer.fc1.weight
                src_layer.fc2.weight = tgt_layer.fc2.weight
                src_layer.self_attn.k_proj.weight = tgt_layer.self_attn.k_proj.weight
                src_layer.self_attn.v_proj.weight = tgt_layer.self_attn.v_proj.weight
                src_layer.self_attn.q_proj.weight = tgt_layer.self_attn.q_proj.weight
                src_layer.self_attn.out_proj.weight = (
                    tgt_layer.self_attn.out_proj.weight
                )
        else:
            if share_level == 1:
                return tgt_layer
        return src_layer

    @classmethod
    def build_spch_encoder(cls, args):
        cfg = {
            "input_feat_per_channel": args.input_feat_per_channel,
            "input_channels": args.input_channels,
            "conv_kernel_sizes": args.conv_kernel_sizes,
            "conv_channels": args.conv_channels,
            "encoder_embed_dim": args.encoder_embed_dim,
            "encoder_ffn_embed_dim": args.encoder_ffn_embed_dim,
            "encoder_layers": args.speech_encoder_layers,
            "encoder_layerdrop": args.encoder_layerdrop,
            "encoder_attention_heads": args.encoder_attention_heads,
            "max_source_positions": args.max_source_positions,
            "dropout": args.dropout,
            "encoder_normalize_before": args.encoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "activation_fn": args.activation_fn,
            "layernorm_embedding": args.layernorm_embedding,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
            "no_scale_embedding": args.no_scale_embedding,
            "quant_noise_pq": args.quant_noise_pq,
            "encoder_freezing_updates": 0,
        }
        model_args = namedtuple("args", cfg.keys())(*cfg.values())
        spch_encoder = S2TTransformerEncoder(model_args)
        if args.add_speech_eos:
            spch_encoder = SpeechEoSEncoder(
                spch_encoder,
                2 * len(args.conv_kernel_sizes.split(",")),
                args.input_feat_per_channel,
                adapter_type=getattr(args, "speech_encoder_adapter_type", "None"),
                adapter_dim=args.encoder_embed_dim,
            )
        return spch_encoder

    @classmethod
    def build_text_encoder(cls, args, src_dictionary, spch_encoder):
        if args.encoder_shared_layers > 0:
            mx_shared_layers = (
                args.speech_encoder_layers
                if args.speech_encoder_layers < args.text_encoder_layers
                else args.text_encoder_layers
            )
            args.encoder_shared_layers = (
                args.encoder_shared_layers
                if args.encoder_shared_layers <= mx_shared_layers
                else mx_shared_layers
            )
        cfg = {
            "encoder_embed_dim": args.encoder_text_embed_dim,
            "encoder_ffn_embed_dim": args.encoder_ffn_embed_dim,
            "encoder_layers": args.text_encoder_layers,
            "encoder_layerdrop": args.encoder_layerdrop,
            "encoder_attention_heads": args.encoder_attention_heads,
            "encoder_learned_pos": args.encoder_learned_pos,
            "max_source_positions": args.max_source_positions,
            "dropout": args.dropout,
            "encoder_normalize_before": args.encoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "activation_fn": args.activation_fn,
            "adaptive_input": args.adaptive_input,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
            "no_scale_embedding": args.no_scale_embedding,
            "quant_noise_pq": args.quant_noise_pq,
        }
        model_args = namedtuple("args", cfg.keys())(*cfg.values())
        enc_emb = nn.Embedding(
            len(src_dictionary), model_args.encoder_embed_dim, src_dictionary.pad()
        )
        text_encoder = TransformerEncoder(model_args, src_dictionary, enc_emb)
        if args.add_speech_eos:
            spch_encoder = spch_encoder.encoder
        if args.encoder_shared_layers > 0:
            text_encoder.layer_norm = cls.set_shared_layer(
                args.encoder_shared_layer_level,
                text_encoder.layer_norm,
                spch_encoder.layer_norm,
            )
            for i, ly in enumerate(
                spch_encoder.transformer_layers[-args.encoder_shared_layers :]
            ):
                ly_id = i + args.text_encoder_layers - args.encoder_shared_layers
                if not isinstance(text_encoder.layers[ly_id], type(ly)):
                    if text_encoder.layers[ly_id]._get_name() not in ('TransformerEncoderLayerBase', 'TransformerEncoderLayer'):
                        raise ValueError("The shared layers are expected from the same class")
                text_encoder.layers[ly_id] = cls.set_shared_layer(
                    args.encoder_shared_layer_level,
                    text_encoder.layers[ly_id],
                    ly,
                )
        return text_encoder

    def mult_rst_grad(self, rst, ratio):
        assert isinstance(rst, dict)  # instead of EncoderOut
        assert len(rst["encoder_out"]) == 1
        rst["encoder_out"][0] = GradMultiply.apply(rst["encoder_out"][0], ratio)
        return rst

    def process_attentive_loss_states(self, rst, interstates):
        assert isinstance(rst, dict)  # instead of EncoderOut
        rst["encoder_states"] = interstates
        return rst

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        src_txt_tokens=None,
        src_txt_lengths=None,
        **kwargs
    ):
        """
        Args:
            src_tokens: padded tensor (B, T, C * feat)
            src_lengths: tensor of original lengths of input utterances (speech) (B,)
            src_txt_tokens: padded tensor (B, T)
            src_txt_lengths: tensor of original lengths of input utterances (text) (B,)
        """
        # src_tokens only: inference
        # src_tokens, src_lengths: speech only training
        # src_txt_tokens, src_txt_lengths: text only training
        # all valid: speech + text training

        if src_tokens is None and src_txt_tokens is None:
            raise ValueError(
                "src_tokens and src_txt_tokens cannot be None at the same time"
            )
        ret1 = None
        ret2 = None
        return_all_hiddens = False
        if src_tokens is not None:
            if (
                self.use_cross_attentive_loss and src_txt_tokens is not None
            ):  # remove self.training so we can get attn score during validation step
                return_all_hiddens = True
            ret1 = self.spch_encoder(
                src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
            )

            if self.use_cross_attentive_loss and src_txt_tokens is not None:
                assert self.cross_attentive_loss_before_last_layer < len(
                    ret1["encoder_states"]
                )
                ret1 = self.process_attentive_loss_states(
                    ret1,
                    ret1["encoder_states"][
                        -self.cross_attentive_loss_before_last_layer - 1
                    ],
                )

        if src_txt_tokens is not None:
            ret2 = self.text_encoder(
                src_txt_tokens, src_txt_lengths, return_all_hiddens=return_all_hiddens
            )
            if return_all_hiddens:
                if self.cross_attentive_loss_before_last_layer == len(
                    self.text_encoder.layers
                ):
                    text_embedding, _ = self.text_encoder.forward_embedding(
                        src_txt_tokens
                    )
                    text_embedding = text_embedding.transpose(0, 1)
                    ret2 = self.process_attentive_loss_states(ret2, text_embedding)
                else:
                    assert self.cross_attentive_loss_before_last_layer < len(
                        self.text_encoder.layers
                    )
                    ret2 = self.process_attentive_loss_states(
                        ret2,
                        ret2["encoder_states"][
                            -self.cross_attentive_loss_before_last_layer - 1
                        ],
                    )

        def merge_output(rst1, rst2):
            if rst1 is None:
                if not (self.enc2_along_grad_mult == 1.0 or self.training):
                    rst2 = self.mult_rst_grad(rst2, self.enc2_along_grad_mult)
                return rst2
            if rst2 is None:
                return rst1
            if self.enc_grad_mult != 1.0 and self.training:
                rst1 = self.mult_rst_grad(rst1, self.enc_grad_mult)
                rst2 = self.mult_rst_grad(rst2, self.enc_grad_mult)
            rst = (rst1, rst2)
            return rst

        return merge_output(ret1, ret2)

    def reorder_encoder_out(self, encoder_out, new_order):
        assert self.training is False  # used for inference only
        return self.spch_encoder.reorder_encoder_out(encoder_out, new_order)


# TransformerMultiInputDecoder: take one or two encoder inputs
class TransformerMultiInputDecoder(FairseqDecoder):
    def __init__(
        self,
        dictionary,
        spch_decoder,
        text_decoder,
        compute_cross_attentive_loss=False,
        cross_attentive_loss_with_norm=True,
        cross_attentive_loss_reverse=False,
    ):

        super().__init__(dictionary)
        self.spch_decoder = spch_decoder
        self.text_decoder = text_decoder
        self.compute_cross_attentive_loss = compute_cross_attentive_loss
        self.cross_attentive_loss_with_norm = cross_attentive_loss_with_norm
        self.cross_attentive_loss_reverse = cross_attentive_loss_reverse

    @classmethod
    def share_spchdecoder(cls, task_args, text_decoder, spch_decoder):
        if task_args.decoder_shared_layer_level == 0:
            return text_decoder
        assert text_decoder.embed_tokens == spch_decoder.embed_tokens
        spch_decoder.project_in_dim = text_decoder.project_in_dim
        spch_decoder.embed_positions = text_decoder.embed_positions
        spch_decoder.layernorm_embedding = text_decoder.layernorm_embedding
        spch_decoder.project_out_dim = text_decoder.project_out_dim
        spch_decoder.adaptive_softmax = text_decoder.adaptive_softmax
        if task_args.decoder_shared_layer_level == 1:
            spch_decoder.output_projection = text_decoder.output_projection
            spch_decoder.layer_norm = text_decoder.layer_norm
        else:  # 2
            spch_decoder.output_projection.weight = (
                text_decoder.output_projection.weight
            )
        for i, ly in enumerate(text_decoder.layers):
            sly = spch_decoder.layers[i]
            sly.self_attn = ly.self_attn
            sly.self_attn_layer_norm = ly.self_attn_layer_norm
            # sly.encoder_attn = ly.encoder_attn
            if (
                task_args.decoder_shared_layer_level == 1
            ):  # share everything, but under different models
                sly.encoder_attn = ly.encoder_attn
                sly.encoder_attn_layer_norm = ly.encoder_attn_layer_norm
                sly.fc1 = ly.fc1
                sly.fc2 = ly.fc2
                sly.final_layer_norm = ly.final_layer_norm
            else:  # task_args.decoder_shared_layer_level == 2: #separated encoder_attn_layer_norm and bias
                sly.encoder_attn.k_proj.weight = ly.encoder_attn.k_proj.weight
                sly.encoder_attn.v_proj.weight = ly.encoder_attn.v_proj.weight
                sly.encoder_attn.q_proj.weight = ly.encoder_attn.q_proj.weight
                sly.encoder_attn.out_proj.weight = ly.encoder_attn.out_proj.weight
                sly.fc1.weight = ly.fc1.weight
                sly.fc2.weight = ly.fc2.weight

        return spch_decoder

    def cross_attentive_loss(
        self, teacher_states, student_states, teacher_masking, student_masking, eps=1e-6
    ):
        x = teacher_states.transpose(0, 1)  # from T X B X D to B X T X D
        y = student_states.transpose(0, 1)
        if self.cross_attentive_loss_with_norm:
            x = x / (x.norm(dim=2, keepdim=True) + eps)
            y = y / (y.norm(dim=2, keepdim=True) + eps)
        dim = x.size(-1)
        # lengths: batch X seqLen
        sim_scores_xy = torch.bmm(x, y.transpose(1, 2))  # batch X lenx X leny ]
        if y.dtype == torch.float16:
            sim_scores_xy = sim_scores_xy.float()
            y = y.float()
            x = x.float()
        if teacher_masking != []:
            assert len(teacher_masking) == 1
            sim_scores_xy = sim_scores_xy.masked_fill(
                teacher_masking[0].unsqueeze(-1), float("-inf")
            )
        if student_masking != []:
            sim_scores_xy = sim_scores_xy.masked_fill(
                student_masking[0].unsqueeze(1), float("-inf")
            )
        # do masking
        y_weights = utils.softmax(sim_scores_xy, dim=-1)
        if teacher_masking != []:
            y_weights = y_weights.masked_fill(teacher_masking[0].unsqueeze(-1), 0)
        x_reconstruct_from_y = torch.bmm(y_weights, y)

        sim_scores_xx = torch.bmm(x, x.transpose(1, 2))  # batch X lenx X lenx ]
        x_weights = utils.softmax(sim_scores_xx, dim=-1)
        if teacher_masking != []:
            x_weights = x_weights.masked_fill(teacher_masking[0].unsqueeze(-1), 0)

        # no gradient for teacher state
        x_reconstruct_from_x = torch.bmm(x_weights, x).detach()
        cost = (x_reconstruct_from_x - x_reconstruct_from_y).norm(dim=2)
        if teacher_masking != []:
            cost = cost.masked_fill(teacher_masking[0], 0)

        if not self.cross_attentive_loss_with_norm:
            cost = cost / dim
        return cost

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state=None,
        has_txt_input=False,
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
                    rst.append(
                        self.spch_decoder(prev_output_tokens, eo, incremental_state)
                    )
                else:
                    rst.append(
                        self.text_decoder(prev_output_tokens, eo, incremental_state)
                    )
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
                    prev_output_tokens, encoder_out, incremental_state
                )
            return self.spch_decoder(prev_output_tokens, encoder_out, incremental_state)


# Note:
# dual input transformer:
#    encoder: S2TTransformerEncoder for speech + TransformerEncoder for text
#    decoder: TransformerDecoder for text
@register_model("dual_input_s2t_transformer")
class DualInputS2TTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.num_updates = 0

    def max_positions(self):
        return None  # it is provided in task

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # encoder 1: S2TTransformerEncoder for speech
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        parser.add_argument(
            "--enc-output-dim",
            type=int,
            metavar="N",
            help="""
                encoder output dimension, can be None. If specified, projecting the
                transformer output to the specified dimension""",
        )
        # standard Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
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
            "--encoder-text-embed-dim",
            type=int,
            metavar="N",
            help="encoder text embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
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
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        # non-standard transformer parameters
        parser.add_argument(
            "--speech-encoder-layers",
            type=int,
            metavar="N",
            help="num speech encoder layers",
        )
        parser.add_argument(
            "--text-encoder-layers",
            type=int,
            metavar="N",
            help="num text encoder layers",
        )
        parser.add_argument(
            "--encoder-shared-layers",
            type=int,
            metavar="N",
            help="num shared encoder layers",
        )
        parser.add_argument(
            "--encoder-shared-layer-level",
            type=int,
            metavar="N",
            default=0,
            choices=[0, 1, 2],
            help="share layer level 0: all share 1: all share with separate model 2: share weight but not bias and layernorm",
        )

        parser.add_argument(
            "--decoder-shared-layer-level",
            default=0,
            choices=[0, 1, 2],
            type=int,
            metavar="N",
            help="0: share everything; 1: share everything with different model 2: no share layer_norm and bias",
        )
        ###
        parser.add_argument(
            "--text-input-cost-ratio",
            type=float,
            default=1.0,
            metavar="V",
            help="text input cost ratio relative to speech input cost",
        )
        parser.add_argument(
            "--init-scale",
            type=float,
            default=1.0,
            metavar="V",
            help="scale the initial weight by given factor",
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
            "--load-pretrain-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained encoder """,
        )
        parser.add_argument(
            "--load-pretrain-speech-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained speech encoder """,
        )
        parser.add_argument(
            "--load-pretrain-text-encoder",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained text encoder """,
        )
        parser.add_argument(
            "--load-pretrain-text-encoder-last",
            type=str,
            default="",
            metavar="EXPR",
            help=""" path to the pretrained text encoder """,
        )
        parser.add_argument(
            "--load-pretrain-decoder",
            type=str,
            metavar="EXPR",
            default="",
            help=""" path to the pretrained encoder """,
        )
        parser.add_argument(
            "--add-speech-eos",
            action="store_true",
            help="add eos token at the end of input feature",
        )
        parser.add_argument(
            "--speech-encoder-adapter-type",
            type=str,
            metavar="EXPR",
            default="None",
            choices=["None", "Linear", "MLP"],
            help="add speech encoder adapter",
        )

    @classmethod
    def build_encoder(cls, args, task):
        spch_encoder = DualInputEncoder.build_spch_encoder(args)
        text_encoder = DualInputEncoder.build_text_encoder(
            args, task.src_dict, spch_encoder
        )
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
        if args.init_scale != 1.0:
            with torch.no_grad():
                for param in encoder.parameters():
                    param.data.mul_(args.init_scale)
        if args.load_pretrain_text_encoder != "":
            checkpoint_utils.load_pretrained_component_from_model(
                text_encoder, args.load_pretrain_text_encoder
            )
        if args.load_pretrain_speech_encoder != "":
            if hasattr(spch_encoder, "encoder"):
                checkpoint_utils.load_pretrained_component_from_model(
                    spch_encoder.encoder, args.load_pretrain_speech_encoder
                )
            else:
                checkpoint_utils.load_pretrained_component_from_model(
                    spch_encoder, args.load_pretrain_speech_encoder
                )
        if (
            args.load_pretrain_text_encoder_last != ""
        ):  # if share encoder, speech encoder parameters will be used.
            # It provides a chance to use pre-trained mt encoder instead
            checkpoint_utils.load_pretrained_component_from_model(
                text_encoder, args.load_pretrain_text_encoder_last
            )

        if args.load_pretrain_encoder != "":
            checkpoint_utils.load_pretrained_component_from_model(
                encoder, args.load_pretrain_encoder
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task):
        dec_cfg = {
            "decoder_layerdrop": args.decoder_layerdrop,
            "share_decoder_input_output_embed": args.share_decoder_input_output_embed,
            "decoder_embed_dim": args.decoder_embed_dim,
            "max_target_positions": args.max_target_positions,
            "dropout": args.dropout,
            "encoder_learned_pos": args.encoder_learned_pos,
            "decoder_learned_pos": args.decoder_learned_pos,
            "layernorm_embedding": args.layernorm_embedding,
            "decoder_normalize_before": args.decoder_normalize_before,
            "activation_dropout": args.activation_dropout,
            "attention_dropout": args.attention_dropout,
            "decoder_ffn_embed_dim": args.decoder_ffn_embed_dim,
            "decoder_layers": args.decoder_layers,
            "decoder_attention_heads": args.decoder_attention_heads,
            "decoder_output_dim": args.decoder_embed_dim,
            "no_scale_embedding": args.no_scale_embedding,
            "adaptive_input": args.adaptive_input,
            "quant_noise_pq": args.quant_noise_pq,
            "adaptive_softmax_cutoff": args.adaptive_softmax_cutoff,
            "tie_adaptive_weights": args.tie_adaptive_weights,
            "no_token_positional_embeddings": args.no_token_positional_embeddings,
        }
        dec_cfg = namedtuple("args", dec_cfg.keys())(*dec_cfg.values())
        dec_emb = nn.Embedding(
            len(task.target_dictionary),
            args.decoder_embed_dim,
            task.target_dictionary.pad(),
        )
        compute_cross_attentive_loss = (
            True if getattr(args, "attentive_cost_regularization", 0.0) > 0.0 else False
        )
        cross_attentive_loss_without_norm = getattr(
            args, "attentive_cost_without_normalize", False
        )
        cross_attentive_loss_reverse = (
            False  # getattr(args, "attentive_cost_reverse", False)
        )

        text_decoder = TransformerDecoder(dec_cfg, task.target_dictionary, dec_emb)
        spch_decoder = TransformerDecoder(dec_cfg, task.target_dictionary, dec_emb)
        spch_decoder = TransformerMultiInputDecoder.share_spchdecoder(
            args, text_decoder, spch_decoder
        )
        decoder = TransformerMultiInputDecoder(
            dictionary=task.target_dictionary,
            spch_decoder=spch_decoder,
            text_decoder=text_decoder,
            compute_cross_attentive_loss=compute_cross_attentive_loss,
            cross_attentive_loss_with_norm=True
            if not cross_attentive_loss_without_norm
            else False,
            cross_attentive_loss_reverse=cross_attentive_loss_reverse,
        )
        if args.init_scale != 1.0:
            with torch.no_grad():
                for param in decoder.parameters():
                    param.data.mul_(args.init_scale)
        if args.load_pretrain_decoder != "":
            try:
                checkpoint_utils.load_pretrained_component_from_model(
                    decoder, args.load_pretrain_decoder
                )
            except RuntimeError:
                checkpoint_utils.load_pretrained_component_from_model(
                    decoder.text_decoder, args.load_pretrain_decoder
                )
                if args.decoder_shared_layer_level > 0:
                    checkpoint_utils.load_pretrained_component_from_model(
                        decoder.spch_decoder, args.load_pretrain_decoder
                    )

        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        dualinputs2ttransformer_base(args)

        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        use_encoder_outputs=False,
        src_txt_tokens=None,
        src_txt_lengths=None,
        mode="sup_speech",
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
        if mode == "text":
            assert src_txt_tokens is None
            src_txt_tokens = src_tokens
            src_txt_lengths = src_lengths
            src_tokens = None
            src_lengths = None
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_txt_tokens=src_txt_tokens,
            src_txt_lengths=src_txt_lengths,
            **kwargs
        )
        has_txt_input = True if src_txt_tokens is not None else False
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            has_txt_input=has_txt_input,
            **kwargs
        )
        if use_encoder_outputs:
            return decoder_out, encoder_out
        return decoder_out


@register_model_architecture(
    "dual_input_s2t_transformer", "dualinputs2ttransformer_base"
)
def dualinputs2ttransformer_base(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_text_embed_dim = getattr(
        args, "encoder_text_embed_dim", args.encoder_embed_dim
    )
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
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
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 10)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.encoder_shared_layers = getattr(args, "encoder_shared_layers", 0)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    args.add_speech_eos = getattr(args, "add_speech_eos", False)


@register_model_architecture("dual_input_s2t_transformer", "dualinputs2ttransformer_s")
def dualinputs2ttransformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 7)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 7)
    args.decoder_layers = getattr(args, "decoder_layers", 7)
    dualinputs2ttransformer_base(args)


@register_model_architecture("dual_input_s2t_transformer", "dualinputs2ttransformer_m")
def dualinputs2ttransformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 10)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    dualinputs2ttransformer_base(args)


@register_model_architecture("dual_input_s2t_transformer", "dualinputs2ttransformer_b")
def dualinputs2ttransformer_b(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.15)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    dualinputs2ttransformer_base(args)


@register_model_architecture("dual_input_s2t_transformer", "dualinputs2ttransformer_l")
def dualinputs2ttransformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    args.speech_encoder_layers = getattr(args, "speech_encoder_layers", 12)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    dualinputs2ttransformer_base(args)
