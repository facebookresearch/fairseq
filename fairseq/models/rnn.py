# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTM, Linear
from fairseq.modules import AdaptiveSoftmax, FairseqDropout


logger = logging.getLogger(__name__)


ATTENTION_TYPES = ["none", "luong-dot", "luong-general", "luong-concat",
                   "bahdanau-dot", "bahdanau-concat", "bahdanau-general", "bahdanau"]


DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@dataclass
class RNNModelConfig(FairseqDataclass):
    # Specific model parameters
    rnn_type: ChoiceEnum(["gru", "lstm"]) = field(
        default="gru",
        metadata={"help": "model type (gru or lstm) for both encoder and decoder"}
    )
    attention_type: ChoiceEnum(ATTENTION_TYPES) = field(
        default="luong-dot",
        metadata={"help": "decoder attention type: can be none, "
                          "luong-dot, luong_concat, luong-general, "
                          "bahdanau-dot, bahdanau-concat, bahdanau-general, "
                          "or the original bahdanau"}
    )
    # Encoder parameters
    encoder_embed_dim: int = field(
        default=512,
        metadata={"help": "encoder embedding dimension"}
    )
    encoder_embed_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to pre-trained encoder embedding"}
    )
    encoder_freeze_embed: bool = field(
        default=False,
        metadata={"help": "freeze encoder embeddings during learning"}
    )
    encoder_hidden_size: int = field(
        default=512,
        metadata={"help": "encoder hidden layer size"}
    )
    encoder_layers: int = field(
        default=1,
        metadata={"help": "number of encoder layers"}
    )
    encoder_bidirectional: bool = field(
        default=False,
        metadata={"help": "make the encoder bidirectional"}
    )
    # Decoder parameters
    decoder_embed_dim: int = field(
        default=512,
        metadata={"help": "decoder embedding dimension"}
    )
    decoder_embed_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to pre-trained decoder embedding"}
    )
    decoder_freeze_embed: bool = field(
        default=False,
        metadata={"help": "freeze decoder embeddings"}
    )
    decoder_hidden_size: int = field(
        default=512,
        metadata={"help": "decoder hidden layer size"}
    )
    decoder_layers: int = field(
        default=1,
        metadata={"help": "number of decoder layers"}
    )
    decoder_out_embed_dim: int = field(
        default=512,
        metadata={"help": "decoder output embedding dimension"}
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default="10000,50000,200000",
        metadata={"help": "comma separated list of adaptive softmax cutoff points. "
                          "Must be used with adaptive_loss criterion"}
    )
    # Embedding management
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"action": "store_true",
                  "help": "share decoder input and output embeddings"}
    )
    share_all_embeddings: bool = field(
        default=False,
        metadata={"action": "store_true",
                  "help": "share encoder, decoder and output embeddings"
                          "(requires shared dictionary and embed dim)"}
    )
    # Dropouts
    dropout: float = field(
        default=0.2,
        metadata={"help": "dropout probability"}
    )
    encoder_dropout_in: float = field(
        default=-1,
        metadata={"help": "dropout probability for encoder input embedding"}
    )
    encoder_dropout_out: float = field(
        default=-1,
        metadata={"help": "dropout probability for encoder output"}
    )
    decoder_dropout_in: float = field(
        default=-1,
        metadata={"help": "dropout probability for decoder input embedding"}
    )
    decoder_dropout_out: float = field(
        default=-1,
        metadata={"help": "dropout probability for decoder output"}
    )


@register_model("rnn", dataclass=RNNModelConfig)
class RNNModel(FairseqEncoderDecoderModel):
    """ Implements a recurrent encoder decoder model, which can use
    uni/bidir GRU/LSTM for the encoder, and unidir GRU/LSTM with
    Luong/Bahdanau/no attention.

    Extends considerably the work done in the lstm enc dec in fairseq.
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @classmethod
    def build_model(cls, cfg: RNNModelConfig, task):
        """Build a new model instance."""
        if cfg.encoder_layers != cfg.decoder_layers:
            raise ValueError("--encoder-layers must match --decoder-layers")

        max_source_positions = getattr(
            cfg, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
        )
        max_target_positions = getattr(
            cfg, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS
        )

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = torch.nn.Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if cfg.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                cfg.encoder_embed_path, task.source_dictionary, cfg.encoder_embed_dim
            )
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = torch.nn.Embedding(
                num_embeddings, cfg.encoder_embed_dim, task.source_dictionary.pad()
            )

        if cfg.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError("--share-all-embeddings requires a joint dictionary")
            if cfg.decoder_embed_path and (
                cfg.decoder_embed_path != cfg.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embed not compatible with --decoder-embed-path"
                )
            if cfg.encoder_embed_dim != cfg.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to "
                    "match --decoder-embed-dim"
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            cfg.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if cfg.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    cfg.decoder_embed_path,
                    task.target_dictionary,
                    cfg.decoder_embed_dim,
                )
        # one last double check of parameter combinations
        if cfg.share_decoder_input_output_embed and (
            cfg.decoder_embed_dim != cfg.decoder_out_embed_dim
        ):
            raise ValueError(
                "--share-decoder-input-output-embeddings requires "
                "--decoder-embed-dim to match --decoder-out-embed-dim"
            )

        if cfg.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if cfg.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = RNNEncoder(
            rnn_type=cfg.rnn_type,
            dictionary=task.source_dictionary,
            embed_dim=cfg.encoder_embed_dim,
            hidden_size=cfg.encoder_hidden_size,
            num_layers=cfg.encoder_layers,
            dropout_in=(cfg.encoder_dropout_in if cfg.encoder_dropout_in >= 0 else cfg.dropout),
            dropout_out=(cfg.encoder_dropout_out if cfg.encoder_dropout_out >= 0 else cfg.dropout),
            bidirectional=cfg.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions,
        )
        uses_attention = getattr(cfg, 'attention_type', "none") != "none"
        attention_type = getattr(cfg, 'attention_type', "luong-dot") if uses_attention else None
        decoder = RNNDecoder(
            rnn_type=cfg.rnn_type,
            dictionary=task.target_dictionary,
            embed_dim=cfg.decoder_embed_dim,
            hidden_size=cfg.decoder_hidden_size,
            out_embed_dim=cfg.decoder_out_embed_dim,
            num_layers=cfg.decoder_layers,
            dropout_in=(cfg.decoder_dropout_in if cfg.decoder_dropout_in >= 0 else cfg.dropout),
            dropout_out=(cfg.decoder_dropout_out if cfg.decoder_dropout_out >= 0 else cfg.dropout),
            attention=uses_attention,
            attention_type=attention_type,
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=cfg.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int)
                if cfg.criterion == "adaptive_loss"
                else None
            ),
            max_target_positions=max_target_positions,
            residuals=False,
        )
        return cls(encoder, decoder)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )
        return decoder_out


class RNNEncoder(FairseqEncoder):
    """RNN encoder."""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        bidirectional=False,
        left_pad_source=True,
        pretrained_embed=None,
        padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
        rnn_type="gru"
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = torch.nn.Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.rnn_type = rnn_type
        if rnn_type == "gru":
            self.hidden = GRU(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=self.dropout_out_module.p if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        elif rnn_type == "lstm":
            self.hidden = LSTM(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=self.dropout_out_module.p if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

        self.left_pad_source = left_pad_source

        self.output_units = hidden_size
        if bidirectional:
            self.bidir_dense = torch.nn.Linear(2, 1)

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        enforce_sorted: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
            enforce_sorted (bool, optional): if True, `src_tokens` is
                expected to contain sequences sorted by length in a
                decreasing order. If False, this condition is not
                required. Default: True.
        """
        if self.left_pad_source:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True,
            )
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.cpu(), enforce_sorted=enforce_sorted,
            batch_first=True
        )

        packed_outs, hidden = self.hidden(packed_x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_idx * 1.0,
            batch_first=True
        )

        if self.bidirectional:
            fwd_final, bwd_final = outputs.view(bsz, max(src_lengths), self.hidden_size, 2).permute(3, 0, 1, 2)
            outputs = torch.cat((fwd_final.unsqueeze(-1), bwd_final.unsqueeze(-1)), -1)
            outputs = self.bidir_dense(outputs).squeeze(-1)

        outputs = self.dropout_out_module(outputs)

        if self.rnn_type == "lstm":
            final_hiddens = self.reshape_state(hidden[0], bsz)
            final_cells = self.reshape_state(hidden[1], bsz)
        else:
            final_hiddens, final_cells = self.reshape_state(hidden, bsz), None

        assert list(outputs.size()) == [bsz, seqlen, self.output_units]

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return tuple(
            (
                outputs,  # batch x seq_len x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden
                final_cells,  # num_layers x batch x num_directions*hidden
                encoder_padding_mask,  # seq_len x batch
            )
        )

    def reshape_state(self, hidden, bsz: int):
        if self.bidirectional:
            fwd_final, bwd_final = hidden.view(2, self.num_layers, bsz, -1)
            last_hidden_state = torch.cat((fwd_final.unsqueeze(-1), bwd_final.unsqueeze(-1)), -1)
            return self.bidir_dense(last_hidden_state).squeeze(-1)
        else:
            return hidden.view(self.num_layers, bsz, -1)

    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple(
            (
                encoder_out[0].index_select(0, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order) if self.rnn_type == "lstm" else None,
                encoder_out[3].index_select(1, new_order),
            )
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class Attention(nn.Module):
    """ Implements the recurrent attentions of
     - Luong, in Effective approaches to attention-based neural machine translation
     - Bahdanau, Neural Machine Translation by Jointly Learning to Align and Translate
     and extends Bahdanau's attentions with Luong's dot/concat/general methods.
    """
    def __init__(self, attention_type, hidden_dimension):
        super().__init__()
        assert attention_type in ATTENTION_TYPES, f"{attention_type}, {ATTENTION_TYPES}"

        self.attention_type = attention_type
        self.hidden_dimension = hidden_dimension
        # Attention
        if self.attention_type in ["none", "luong-dot", "bahdanau-dot"]:
            pass
        elif self.attention_type in ["luong-general", "bahdanau-general"]:
            self.Wa = torch.nn.Linear(self.hidden_dimension, self.hidden_dimension, bias=False)
        elif self.attention_type in ["luong-concat", "bahdanau-concat"]:
            self.Wa = torch.nn.Linear(2 * self.hidden_dimension, self.hidden_dimension, bias=False)
            self.va = torch.nn.Parameter(torch.randn(1, self.hidden_dimension), requires_grad=True)
        elif self.attention_type == "bahdanau":
            self.Wa = torch.nn.Linear(self.hidden_dimension, self.hidden_dimension, bias=False)
            self.Ua = torch.nn.Linear(self.hidden_dimension, self.hidden_dimension, bias=False)
            self.va = torch.nn.Parameter(
                torch.randn(1, 2 * self.hidden_dimension), requires_grad=True
            )
        else:
            raise logger.error("Bad attention type given to the decoder", "USER_ERROR")

        if self.attention_type in ["none", "luong-dot", "luong-general", "luong-concat"]:
            self.Wc = torch.nn.Linear(self.hidden_dimension * 2, self.hidden_dimension)

    def forward(self, encoder_output: torch.Tensor,
        prev_prediction_emb: torch.Tensor = None,
        prev_hidden_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        target_state: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attention_type in ["luong-dot", "luong-general", "luong-concat"]:
            return self.luong_forward(target_state, encoder_output)
        if self.attention_type in ["bahdanau-dot", "bahdanau-concat", "bahdanau-general", "bahdanau"]:
            return self.bahdanau_forward(prev_prediction_emb, encoder_output, prev_hidden_state)
        return target_state, None

    def luong_forward(self, target_state, encoder_output):
        attn_score = self.get_attention_score(target_state, encoder_output)
        context, softmax_attn = self.compute_context(attn_score, encoder_output)
        output = self.compute_attentional_hidden_state(target_state, context)
        return output, softmax_attn

    def bahdanau_forward(self, prev_prediction_emb, encoder_output, prev_hidden_state):
        bsz = encoder_output.shape[0]
        # We take the last layer of the hidden state
        attn_score = self.get_attention_score(prev_hidden_state[-1:], encoder_output)
        context, softmax_attn = self.compute_context(attn_score, encoder_output)
        concat_context_embed = torch.cat(
            (prev_prediction_emb.view(bsz, 1, -1), context), 2
        ).transpose(0, 1)
        return concat_context_embed, softmax_attn

    def get_attention_score(
        self, source_or_target_state: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """ Computes Luong's or Bahdanau's attention.
        This either follows the equations of Effectives approaches to AttentionBased Neural Machine Translation (Luong)
        or Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau)
        """
        # DOT
        if self.attention_type in ["luong-dot", "bahdanau-dot"]:
            state_transposed = source_or_target_state.squeeze(0).unsqueeze(-1)  # transpose ht
            attn_score = torch.bmm(encoder_output, state_transposed).squeeze(-1)  # dot product
        # GENERAL
        elif self.attention_type in ["luong-general", "bahdanau-general"]:
            state_transposed = source_or_target_state.squeeze(0).unsqueeze(-1)  # transpose ht
            attn_score = torch.bmm(self.Wa(encoder_output), state_transposed).squeeze(
                -1
            )
        # CONCAT
        elif self.attention_type in ["luong-concat", "bahdanau-concat"]:
            # we apply the current ht to all elements of hs
            extended = source_or_target_state.repeat(encoder_output.shape[1], 1, 1)
            concat = torch.cat((extended.transpose(0, 1), encoder_output), -1)
            wa = torch.tanh(self.Wa(concat))
            va = self.va.repeat(encoder_output.shape[0], 1).unsqueeze(-1)
            attn_score = torch.bmm(wa, va).squeeze(-1)
        # ORIGINAL BAHDANAU
        elif self.attention_type == "bahdanau":
            # we apply the current ht to all elements of hs
            #print(encoder_output.shape, source_or_target_state.shape)
            extended = source_or_target_state.repeat(encoder_output.shape[1], 1, 1)
            #print(extended.shape, self.Wa(extended.transpose(0, 1)).shape, self.Ua(encoder_output).shape)
            concat = torch.cat((self.Wa(extended.transpose(0, 1)), self.Ua(encoder_output)), -1)
            wa = torch.tanh(concat)
            va = self.va.repeat(encoder_output.shape[0], 1).unsqueeze(-1)
            attn_score = torch.bmm(wa, va).squeeze(-1)
        # OTHER
        else:
            raise logger.error(f"Wrong attention type ({self.attention_type.name})")

        return attn_score

    def compute_context(
        self, attn_score: torch.Tensor, encoder_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes the context, depending on the type of attention """
        softmax_attn = torch.nn.functional.softmax(attn_score, dim=1).unsqueeze(1)  # alignment vector alpha
        if self.attention_type in ["luong-dot", "luong-general", "luong-concat"]:
            cur_c = torch.bmm(softmax_attn, encoder_output).transpose(0, 1) # (B, 1, H) to (1, B, H)  # ci
        elif self.attention_type in ["bahdanau-dot", "bahdanau-concat", "bahdanau-general", "bahdanau"]:
            cur_c = torch.bmm(softmax_attn, encoder_output)  # ci
        else:
            raise logger.error(f"Wrong attention type ({self.attention_type.name})")
        return cur_c, softmax_attn.squeeze(1)

    def compute_attentional_hidden_state(
        self, hidden_state: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """ For Luong, computes the attentional hidden state """
        concat_ctx_hidden = torch.cat((context.squeeze(0), hidden_state.squeeze(0)), 1)  # (B, 2H)
        concat_layer_ctx_hidden = self.Wc(torch.tanh(concat_ctx_hidden)).unsqueeze(0)  # (1, B, H)
        attn_hidden_state = torch.nn.functional.relu(concat_layer_ctx_hidden)  # Is softmax in orig eq, could be tanh...
        return attn_hidden_state


class RNNDecoder(FairseqIncrementalDecoder):
    """RNN decoder."""

    def __init__(
        self,
        dictionary,
        rnn_type="lstm",
        embed_dim=512,
        hidden_size=512,
        out_embed_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        attention_type="luong-dot",
        encoder_output_units=512,
        pretrained_embed=None,
        share_input_output_embed=False,
        adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        residuals=False,
    ):
        super().__init__(dictionary)
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out, module_name=self.__class__.__name__
        )
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = torch.nn.Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = torch.nn.Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = torch.nn.Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        # For Bahdanau, we compute the context on the input feed
        bahd_factor = hidden_size \
            if attention_type in ["bahdanau-dot", "bahdanau-concat", "bahdanau-general", "bahdanau"] \
            else 0
        self.rnn_type = rnn_type
        if rnn_type == "lstm":
            self.layers = LSTM(
                input_size=input_feed_size + embed_dim + bahd_factor,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
        else:
            self.layers = GRU(
                input_size=input_feed_size + embed_dim + bahd_factor,
                hidden_size=hidden_size,
                num_layers=num_layers
            )

        if attention_type == "none":
            self.attention_type = "none"
            self.attention = None
        else:
            self.attention_type = attention_type
            self.attention = Attention(self.attention_type, hidden_size)

        if hidden_size != out_embed_dim:
            self.additional_fc = torch.nn.Linear(hidden_size, out_embed_dim)

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings,
                hidden_size,
                adaptive_softmax_cutoff,
                dropout=dropout_out,
            )
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        src_lengths: Optional[Tensor] = None,
    ):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features. Used in beam inference
        """
        # Todo: manage encoder_padding_mask
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out[0]
            encoder_hiddens = encoder_out[1]
            encoder_cells = encoder_out[2]
            encoder_padding_mask = encoder_out[3]
        else:
            encoder_outs = torch.empty(0)
            encoder_hiddens = torch.empty(0)
            encoder_cells = torch.empty(0)
            encoder_padding_mask = torch.empty(0)

        srclen = encoder_outs.size(1)

        # In beam
        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed target tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)
        x = x.transpose(0, 1)

        # Initialize previous states
        ## From cache
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state
            )
        ## From target
        elif encoder_out is not None:
            # setup recurrent cells
            prev_hiddens = torch.stack([encoder_hiddens[i] for i in range(self.num_layers)])
            if encoder_cells is not None:
                prev_cells = torch.stack([encoder_cells[i] for i in range(self.num_layers)])
            else:
                prev_cells = torch.stack([x.new_zeros(bsz, self.hidden_size) for _ in range(self.num_layers)])
            if self.encoder_hidden_proj is not None:
                prev_hiddens = self.encoder_hidden_proj(prev_hiddens)
                prev_cells = self.encoder_cell_proj(prev_cells)
            input_feed = x.new_zeros(bsz, self.hidden_size)
        ## No encoder
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = torch.stack([zero_state for _ in range(self.num_layers)])
            prev_cells = torch.stack([zero_state for _ in range(self.num_layers)])
            input_feed = None

        if self.rnn_type == "gru":
            hidden = prev_hiddens
        else:
            hidden = (prev_hiddens, prev_cells)

        assert (
            srclen > 0 or self.attention is None
        ), "attention is not supported if there are no encoder outputs"
        attn_scores = (
            x.new_zeros(bsz, srclen, seqlen) if self.attention is not None else None
        )  # to store attentions

        outs = []
        # We use 100% teacher forcing
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            # recurrent cell
            if self.attention_type in ["luong-dot", "luong-general", "luong-concat"]:
                # In Luong's attention, you first go through the rnn,
                # then compute the attention score on its output
                assert attn_scores is not None
                target_state, hidden = self.layers(input.unsqueeze(0), hidden)
                # todo: add encoder_padding_mask to attention
                out, attn = self.attention(encoder_outs, target_state=target_state)
                attn_scores[:, :, j] = attn

            elif self.attention_type in ["bahdanau-dot", "bahdanau-concat", "bahdanau-general", "bahdanau"]:
                assert attn_scores is not None
                # We compute the attention on the
                concat_context_embed, attn_scores[:, :, j] = self.attention(
                    encoder_outs,
                    prev_prediction_emb=input,
                    prev_hidden_state=prev_hiddens,
                )
                # todo: add encoder_padding_mask to attention
                out, hidden = self.layers(concat_context_embed, hidden)
            else:
                # No attention here
                out, hidden = self.layers(input.unsqueeze(0), hidden)

            # save state for next time step
            if self.rnn_type == "gru":
                prev_hiddens, prev_cells = hidden, None
            else:
                prev_hiddens, prev_cells = hidden[0], hidden[1]

            # apply attention using the last layer's hidden state
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out.squeeze(0)

            # save final output
            outs.append(out)

        # Store the cache
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens,
                "prev_cells": prev_cells,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, "additional_fc") and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def get_cached_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        cached_state = self.get_incremental_state(incremental_state, "cached_state")
        assert cached_state is not None
        prev_hiddens = cached_state["prev_hiddens"]
        assert prev_hiddens is not None
        prev_cells = cached_state["prev_cells"]
        if self.rnn_type == "lstm":
            assert prev_cells is not None
        input_feed = cached_state[
            "input_feed"
        ]  # can be None for decoder-only language models
        return prev_hiddens, prev_cells, input_feed

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        if self.rnn_type == "lstm":
            prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells) if self.rnn_type == "lstm" else None,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new),
        return

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def GRU(input_size, hidden_size, **kwargs):
    m = nn.GRU(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def GRUCell(input_size, hidden_size, **kwargs):
    m = nn.GRUCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('rnn', 'lstm2')
def lstm(cfg: RNNModelConfig):
    cfg.encoder_bidirectional = False
    cfg.rnn_type = "lstm"


@register_model_architecture('rnn', 'gru')
def gru(cfg: RNNModelConfig):
    cfg.encoder_bidirectional = False
    cfg.rnn_type = "gru"


@register_model_architecture('rnn', 'bilstm')
def bilstm(cfg: RNNModelConfig):
    cfg.encoder_bidirectional = True
    cfg.rnn_type = "lstm"


@register_model_architecture('rnn', 'bigru')
def bigru(cfg: RNNModelConfig):
    cfg.encoder_bidirectional = True
    cfg.rnn_type = "gru"
