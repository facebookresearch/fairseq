# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import (
    LSTMModel,
    LSTMEncoder,
    LSTMDecoder,
    Embedding,
    LSTMCell,
    Linear,
)
from fairseq.modules import AdaptiveSoftmax

from examples.simultaneous_translation.utils.functions import (
    exclusive_cumprod,
    moving_sum,
    lengths_to_mask,
)


@register_model('lstm_monotonic')
class LSTMMonotonicAttentionModel(LSTMModel):
    """
    Implementing three monotonic attention style LSTM models
    Online and Linear-Time Attention by Enforcing Monotonic Alignments
    (https://arxiv.org/pdf/1704.00784.pdf)
    Monotonic Chunk Attention
    (https://arxiv.org/pdf/1712.05382.pdf) (MoChA)
    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    (https://arxiv.org/pdf/1906.05218.pdf) (MILK)(WIP)
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        super(LSTMMonotonicAttentionModel, LSTMMonotonicAttentionModel).add_args(parser)
        # monotonic attention
        parser.add_argument("--monotonic-attention", choices=["hard", "chunk", "infinite"],
                            help="Use monotonic attention, choose from [hard, chunck, infinite]")
        parser.add_argument("--mocha-chunk-size", type=int,
                            help="Chunk size used for mocha")
        parser.add_argument("--no-denominator", action="store_true",
                            help="Set denominator to 1 when calculating attention")
        parser.add_argument("--eps", type=float,
                            help="Epsilon used for preventin underflow")
        parser.add_argument("--bias-init", type=float,
                            help="Default initial bias for energy function")
        parser.add_argument("--scale", action="store_true",
                            help="Scale the energy function")
        parser.add_argument("--attention-type", type=str, choices=["add", "prod"],
                            help="Additive attention or product attention")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = LSTMMonotonicEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        decoder = LSTMMonotonicDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            monotonic_attention=args.monotonic_attention,
            no_denominator=args.no_denominator,
            eps=args.eps,
            bias_init=args.bias_init,
            scale=args.scale,
            mocha_chunk_size=args.mocha_chunk_size,
            attention_type=args.attention_type,
        )
        return cls(encoder, decoder)


class LSTMMonotonicEncoder(LSTMEncoder):
    """LSTM encoder."""

    def reorder_encoder_out(self, encoder_out, new_order):
        for key in encoder_out.keys():
            if encoder_out[key] is None:
                continue
            if key == 'encoder_out':
                encoder_out[key] = tuple(
                    eo.index_select(1, new_order)
                    for eo in encoder_out[key]
                )
            else:
                encoder_out[key] = encoder_out[key].index_select(1, new_order)

        return encoder_out


class EnergyLayer(nn.Module):
    """
    Module to calculate unormalized enegries.
    The energy then can be used to calculate softmax or monotonic attention.
    """
    def __init__(self, input_dim, context_dim, attention_dim=None, bias=False, scale=False, bias_init=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.bias = bias
        self.scale = scale
        self.bias_init = bias_init
        if attention_dim is None:
            self.attention_dim = self.input_dim
        else:
            self.attention_dim = attention_dim

        self._init_modules()

    def _init_modules(self):
        raise NotImplementedError

    def forward(self, input, source_hids, encoder_padding_mask, exponential=False):
        energy = self.calculate_energy(input, source_hids, encoder_padding_mask)
        if exponential:
            energy = torch.exp(energy)
        return energy

    def calculate_energy(self, input, source_hids, encoder_padding_mask):
        raise NotImplementedError


class AdditiveEnergyLayer(EnergyLayer):
    def _init_modules(self):
        if self.attention_dim is None:
            self.attention_dim = self.context_dim
        # W_ae and b_a
        self.encoder_proj = nn.Linear(self.context_dim, self.attention_dim, bias=True)
        # W_ad
        self.decoder_proj = nn.Linear(self.input_dim, self.attention_dim, bias=False)
        # V_a
        if self.scale:
            self.to_scores = nn.utils.weight_norm(
                nn.Linear(self.attention_dim, 1, bias=False)
            )
            self.r = nn.Parameter(torch.ones(1) * self.bias_init)
        else:
            self.to_scores = nn.Linear(self.attention_dim, 1, bias=False)
            self.r = 0

    def calculate_energy(self, input, source_hids, encoder_padding_mask):
        """
        input: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        """
        _, input_dim = input.size()
        src_len, bsz, context_dim = source_hids.size()
        # (src_len*bsz) x context_dim (to feed through linear)
        flat_source_hids = source_hids.view(-1, self.context_dim)
        # (src_len*bsz) x attention_dim
        encoder_component = self.encoder_proj(flat_source_hids)
        # src_len x bsz x attention_dim
        encoder_component = encoder_component.view(src_len, bsz, self.attention_dim)
        # 1 x bsz x attention_dim
        decoder_component = self.decoder_proj(input).unsqueeze(0)
        # Sum with broadcasting and apply the non linearity
        # src_len x bsz x attention_dim
        hidden_att = torch.tanh(
            (decoder_component + encoder_component).view(-1, self.attention_dim)
        )
        # Project onto the reals to get attentions scores (src_len x bsz)
        energy = self.to_scores(hidden_att).view(src_len, bsz)

        # Mask + softmax (src_len x bsz)
        if encoder_padding_mask is not None:
            energy = (
                energy.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(energy)
            )  # FP16 support: cast to float and back

        if self.scale:
            energy += self.r

        return energy


class ProductEnergyLayer(EnergyLayer):
    def _init_modules(self):
        self.input_proj = Linear(self.input_dim, self.context_dim, bias=self.bias)

        if self.scale:
            self.scale_value = nn.Parameter(torch.rsqrt(torch.ones(1) * self.input_dim))
            self.r = nn.Parameter(torch.ones(1) * self.bias_init)
        else:
            self.scale_value = 1
            self.r = 0

    def calculate_energy(self, input, source_hids, encoder_padding_mask):
        x = self.input_proj(input)
        energy = (source_hids * x.unsqueeze(0)).sum(dim=2)
        # Scale energy if needed
        energy = self.scale_value * energy + self.r

        if encoder_padding_mask is not None:
            energy = energy.float().masked_fill_(encoder_padding_mask, float('-inf')).type_as(energy)

        return energy


class MonotonicAttentionLayer(nn.Module):
    """MonotonicAttention from
        'Online and Linear-Time Attention by Enforcing Monotonic Alignments'
        Colin Raffel et al. (2017)
        https://arxiv.org/pdf/1704.00784.pdf

        e_ij = energy_funtion(s_i, h_j)
        p_ij = sigmoid(e_ij)

        at test time
        z_ij ~ Bernoulli(p_ij)

        at training time
        q_ij = (1 − p_{ij−1})q_{ij−1} + a+{i−1j}
        a_ij = p_ij q_ij

        parellel solution:
        ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))
    """
    def __init__(
        self,
        decoder_hidden_state_dim: int,
        encoder_hidden_state_dim: int,
        output_dim: int,
        bias: bool = False,
        monotonic_type: str = "hard",
        bias_init: float = -4.0,
        no_denominator: bool = False,
        scale: bool = False,
        eps: float = 1e-8,
        attention_type: str = "prod",
        mocha_chunk_size: int = 0,
    ):
        super().__init__()

        self.eps = eps
        self.no_denominator = no_denominator
        self.attention_type = attention_type
        self.bias_init = bias_init
        self.bias = bias
        self.scale = scale
        if self.attention_type == "add":
            self.energy_type = AdditiveEnergyLayer
        elif self.attention_type == "prod":
            self.energy_type = ProductEnergyLayer
        else:
            raise RuntimeError("Please choose either add or prod attention.")

        # Monotonic energy function
        self.monotonic_energy_layer = self.energy_type(
            decoder_hidden_state_dim,
            encoder_hidden_state_dim,
            scale=self.scale,
            bias=self.bias,
            bias_init=self.bias_init,
        )

        self.monotonic_type = monotonic_type
        self.mocha_chunk_size = mocha_chunk_size
        if self.monotonic_type != "hard":
            if self.monotonic_type == "chunk" and self.mocha_chunk_size == 0:
                raise RuntimeError("Must choose a window size for mocha model!")
            self.softattn_energy_layer = self.energy_type(
                decoder_hidden_state_dim,
                encoder_hidden_state_dim,
                scale=False,
                bias=self.bias,
                bias_init=0,
            )

        self.output_proj = Linear(
            decoder_hidden_state_dim + encoder_hidden_state_dim,
            output_dim,
            bias=bias
        )

    def forward(
        self,
        input,
        source_hids,
        encoder_padding_mask,
        previous_attention=None,
        monotonic_step=None,
    ):
        """The expected input dimensions are:
        input: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        previous_attention: src_len x bsz
        monotonic_attention_mask: src_len x bsz,
        """
        if monotonic_step is None:
            attn_scores, monotonic_scores = self.attn_scores_train(
                input, source_hids, encoder_padding_mask, previous_attention
            )
        else:
            attn_scores = self.attn_scores_infer(input, source_hids, encoder_padding_mask, monotonic_step)
            monotonic_scores = None

        # Sum weighted sources (bsz x context_dim)
        weighted_context = (
            source_hids
            * attn_scores.type_as(source_hids).unsqueeze(2)
        ).sum(dim=0)

        output = torch.tanh(self.output_proj(torch.cat([weighted_context, input], dim=1)))

        return output, attn_scores, monotonic_scores

    def attn_scores_infer(
        self,
        input,
        source_hids,
        encoder_padding_mask,
        monotonic_step=None,
    ):
        """The expected input dimensions are:
        input: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        monotonic_step: src_len x bsz
        """
        if self.monotonic_type == "hard":
            attn_scores = source_hids.new_zeros(
                [source_hids.size(0), source_hids.size(1)]
            ).scatter(0, monotonic_step, 1)
        else:
            exp_softattn_energy = self.softattn_energy_layer(
                input, source_hids, encoder_padding_mask, exponential=True
            )

            max_src_len = source_hids.size(0)
            monotonic_mask = self.monotonic_mask_from_step(monotonic_step, max_src_len)

            masked_exp_softattn_energy = exp_softattn_energy * monotonic_mask.type_as(exp_softattn_energy)
            attn_scores = masked_exp_softattn_energy / masked_exp_softattn_energy.sum(dim=0)

        return attn_scores

    def attn_scores_train(
        self,
        input,
        source_hids,
        encoder_padding_mask,
        previous_attention=None,
    ):
        """The expected input dimensions are:
        input: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        previous_attention: src_len x bsz
        """
        if previous_attention is None:
            previous_attention = source_hids.new_zeros(
                [
                    source_hids.size(0),
                    source_hids.size(1)
                ]
            )
            previous_attention[0] = 1
        else:
            previous_attention = previous_attention.squeeze(1)

        # src_len x bsz
        p_choose = self.p_choose(input, source_hids, encoder_padding_mask)

        # src_len x bsz
        cumprod_one_minus_p_choose = exclusive_cumprod(1 - p_choose, dim=0, eps=self.eps)

        if self.no_denominator:
            denominator = 1.0
        else:
            denominator = torch.clamp(cumprod_one_minus_p_choose, self.eps, 1.0)

        alpha = (
            p_choose
            * cumprod_one_minus_p_choose
            * torch.cumsum(
                previous_attention / denominator,
                dim=0,
            )
        )


        if self.monotonic_type != "hard":
            """
            Caculate soft energy and expected attention
            """
            softattn_energy = self.softattn_energy_layer(
                input, source_hids, encoder_padding_mask, exponential=False
            )

            # To prevent numerical issue
            softattn_energy_max, _ = torch.max(softattn_energy, dim=0)
            exp_softattn_energy = torch.exp(softattn_energy - softattn_energy_max)

            if self.monotonic_type == "chunk":
                beta = self.mocha_expected_attention(alpha, exp_softattn_energy)
            else:
                beta = self.milk_expected_attention(alpha, exp_softattn_energy)
        else:
            beta = alpha

        return beta, alpha

    def mocha_expected_attention(self, alpha, exp_softattn_energy):
        """
        Monotonic Chunkwise Attention (MoCha)
        """
        beta = (
            exp_softattn_energy
            * moving_sum(
                alpha / (self.eps + moving_sum(exp_softattn_energy, self.mocha_chunk_size, 1)),
                1, self.mocha_chunk_size
            )
        )
        return beta

    def milk_expected_attention(self, alpha, exp_softattn_energy):
        """
        Monotonic Infinite LookbacK attention (MILK)
        """
        beta = (
            exp_softattn_energy
            * moving_sum(
                alpha / (self.eps + torch.cumsum(exp_softattn_energy, dim=0)),
                1, exp_softattn_energy.size(0)
            )
        )
        return beta

    def monotonic_mask_from_step(self, monotonic_step, max_src_len):
        """
        Convert a tensor of lengths to mask

        Also, consider chunk size is needed
        For example, lengths = [[2, 3, 4]], max_len = 5, window_size=2
        mask =
           [[1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]]
        """
        monotonic_mask = lengths_to_mask(monotonic_step + 1, max_src_len)

        if self.monotonic_type == "chunk":
            start_step = monotonic_step - self.mocha_chunk_size + 1
            start_step[start_step < 0] = 0
            chunk_mask = lengths_to_mask(start_step, max_src_len, negative_mask=True)
            monotonic_mask *= chunk_mask

        return monotonic_mask

    def p_choose(self, decoder_state, source_hids, encoder_padding_mask=None):
        """The expected input dimensions are:
        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        """
        energy = self.monotonic_energy_layer(decoder_state, source_hids, encoder_padding_mask)
        # src_len x bsz x 1
        if self.training:
            noise = torch.normal(0.0, 1.0, energy.size()).type_as(energy).to(energy.device)
            p_choose = torch.sigmoid(energy + noise)
        else:
            p_choose = torch.sigmoid(energy)

        return p_choose

    def update_monotonic_step(
        self, decoder_state, source_hids, encoder_padding_mask, prev_monotonic_step
    ):
        """The expected input dimensions are:
        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        prev_monotonic_step: 1 x bsz,
        """
        src_len, bsz, _ = source_hids.size()
        # 1 x bsz
        if encoder_padding_mask is not None:
            src_lengths = src_len - encoder_padding_mask.type_as(prev_monotonic_step).sum(dim=0, keepdim=True)
        else:
            src_lengths = src_len * torch.ones_like(prev_monotonic_step)
        # 1 x bsz
        new_monotonic_step = prev_monotonic_step
        # 1 x bsz
        finish_read = new_monotonic_step.eq(src_lengths - 1).long()

        while finish_read.sum().item() < bsz:
            # 1 x bsz
            p_choose = self.p_choose(
                decoder_state,
                source_hids.gather(
                    0,
                    new_monotonic_step.unsqueeze(2).expand(1, bsz, source_hids.size(2)),
                ),
            )
            action = (p_choose > 0.5).type_as(finish_read) * (1 - finish_read)
            # dist = torch.distributions.bernoulli.Bernoulli(p_choose)
            # 1 x bsz
            # sample actions on unfinished seq
            # 1 means stay, finish reading
            # 0 means leave, continue reading
            # action = dist.sample().type_as(finish_read) * (1 - finish_read)

            new_monotonic_step += (1 - action) * (1 - finish_read)

            finish_read += action

            # make sure don't exceed the source length
            finish_read += new_monotonic_step.eq(src_lengths - 1).long()

            finish_read = 1 - finish_read.eq(0).long()

        return new_monotonic_step


class LSTMMonotonicDecoder(LSTMDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        monotonic_attention=None, no_denominator=False,
        eps=1e-8, bias_init=-4.0, scale=False, mocha_chunk_size=None,
        attention_type="prod",
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        self.monotonic_attention = monotonic_attention
        self.eps = eps
        self.no_denominator = no_denominator
        self.scale = scale
        self.bias_init = bias_init
        self.mocha_chunk_size = mocha_chunk_size
        self.attention_type = attention_type

        self.attention = MonotonicAttentionLayer(
            hidden_size,
            encoder_output_units,
            hidden_size,
            bias=False,
            monotonic_type=self.monotonic_attention,
            no_denominator=self.no_denominator,
            scale=self.scale,
            bias_init=bias_init,
            eps=self.eps,
            mocha_chunk_size=mocha_chunk_size,
            attention_type=attention_type
        )

        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        encoder_out_dict = encoder_out
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']
        encoder_out = encoder_out_dict['encoder_out']
        monotonic_step = encoder_out_dict.get('monotonic_step', None)
        monotonic_step_buffer = encoder_out_dict.get('monotonic_step_buffer', None)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if monotonic_step is None:
                monotonic_step = torch.zeros_like(prev_output_tokens).t()
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        attn_scores_list = [None]
        monotonic_scores_list = []
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                if monotonic_step is not None:
                    monotonic_step = self.attention.update_monotonic_step(
                        hidden, encoder_outs, encoder_padding_mask, monotonic_step)
                out, attn_scores, monotonic_scores = self.attention(
                    hidden,
                    encoder_outs,
                    encoder_padding_mask,
                    previous_attention=attn_scores_list[-1],
                    monotonic_step=monotonic_step
                )
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)
            attn_scores_list.append(attn_scores.unsqueeze(1))
            if monotonic_scores is not None:
                monotonic_scores_list.append(monotonic_scores.unsqueeze(1))

        if monotonic_step is not None:
            encoder_out_dict["monotonic_step"] = monotonic_step

            if monotonic_step_buffer is None:
                encoder_out_dict["monotonic_step_buffer"] = monotonic_step
            else:
                encoder_out_dict["monotonic_step_buffer"] = torch.cat(
                    [
                        monotonic_step_buffer,
                        monotonic_step
                    ],
                    dim=0
                )
                if encoder_padding_mask is not None:
                    src_lens = (1 - encoder_padding_mask.long()).sum(dim=0)
                else:
                    src_lens = encoder_outs.new_ones(1) * encoder_outs.size(0)

                encoder_out_dict["average_proportion"] = AverageProportion(
                    encoder_out_dict["monotonic_step_buffer"],
                    src_lens,
                )

                encoder_out_dict["average_lagging"] = AverageLagging(
                    encoder_out_dict["monotonic_step_buffer"],
                    src_lens,
                )
                encoder_out_dict["differentiable_average_lagging"] = DifferentiableAverageLagging(
                    encoder_out_dict["monotonic_step_buffer"],
                    src_lens,
                )
            monotonic_scores = None
        else:
            monotonic_scores = torch.cat(
                monotonic_scores_list,
                dim=1
            )

        attn_scores = torch.cat(attn_scores_list[1:], dim=1)

        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, 'additional_fc'):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)

        if monotonic_scores is not None:
            latency_term = self.latency_term(
                monotonic_scores,
                encoder_padding_mask,
                prev_output_tokens
                )
        else:
            latency_term = None

        return x, attn_scores, latency_term

    def latency_term(self, alpha, encoder_padding_mask, target_tokens):
        """
        Caculating latency term given expected alignment alpha
        alpha: src_len x tgt_len x bsz
        """
        src_len, tgt_len, bsz = alpha.size()
        tgt_len -= 1
        tgt_pad_mask = target_tokens[:, 1:].eq(1).t()
        # A hack here some times the target tokens contain only BOS
        tgt_pad_mask[0] = False

        expected_latency = (
            torch
            .arange(1, src_len + 1)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(src_len, tgt_len, bsz)
            .type_as(alpha)
            * alpha[:, 1:, :]
        ).sum(dim=0)

        if encoder_padding_mask is not None:
            src_lens = (1 - encoder_padding_mask.type_as(alpha)).sum(dim=0)
        else:
            src_lens = alpha.new_ones(alpha.size(2)) * alpha.size(0)

        latency_term = DifferentiableAverageLagging(
            expected_latency,
            src_lens,
            tgt_pad_mask,
        )
        return latency_term


@register_model_architecture('lstm_monotonic', 'lstm_monotonic')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')

    args.no_denominator = getattr(args, "no_denominator", False)
    args.eps = getattr(args, 'eps', 1e-8)
    args.bias_init = getattr(args, 'bias_init', -4.0)
    args.scale = getattr(args, "scale", False)
    args.mocha_chunk_size = getattr(args, "mocha_chunk_size", 0)
    args.attention_type = getattr(args, "attention_type", "prod")


@register_model_architecture('lstm_monotonic', 'lstm_monotonic_iwslt2015_en_vi')
def lstm_monotonic_iwslt2015_en_vi(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.monotonic_attention = getattr(args, "monotonic_attention", "hard")
    args.no_denominator = getattr(args, "no_denominator", False)
    args.eps = getattr(args, 'eps', 1e-6)
    args.bias_init = getattr(args, 'bias_init', -2.0)
    args.scale = getattr(args, "scale", True)
    args.mocha_chunk_size = getattr(args, "mocha_chunk_size", 0)
    args.attention_type = getattr(args, "attention_type", "prod")
