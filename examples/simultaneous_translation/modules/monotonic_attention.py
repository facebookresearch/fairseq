import torch
from torch import nn
from fairseq import utils

from . import register_monotonic_attention
from .energy_layers import AdditiveEnergyLayer
from examples.simultaneous_translation.utils.functions import (
    exclusive_cumprod
)

from fairseq.incremental_decoding_utils import with_incremental_state
@with_incremental_state
class MonotonicAttentionLayer(nn.Module):
    """
    General template for monotonic attention model
    It uses a expected alignment from stepwise decision for training

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
    a_i = p_i * cumprod(1 − p_i) * cumsum(a_{i-1} / cumprod(1 − p_i))
    """
    def __init__(self, args):
        super().__init__()
        self.mass_preservation = args.mass_preservation
        self.scale_monotonic_energy = args.scale_monotonic_energy
        self.init_monotonic_bias = args.init_monotonic_bias
        self.noise_var = args.noise_var
        self.noise_avg = args.noise_avg
        self.eps = args.monotonic_eps
        self.decoder_hidden_dim = args.decoder_hidden_dim
        self.encoder_hidden_dim = args.encoder_hidden_size
        self.output_dim = args.decoder_hidden_dim

        self.monotonic_energy_layer = AdditiveEnergyLayer(
            self.decoder_hidden_dim,
            self.encoder_hidden_dim,
            scale=self.scale_monotonic_energy,
            init_bias=self.init_monotonic_bias
        )

        self.target_step = 0

        # No output projection in berard?
        # self.output_proj = torch.Linear(
        #     decoder_hidden_state_dim + encoder_hidden_state_dim,
        #     output_dim,
        #     bias=bias
        # )

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--mass-preservation",
            action="store_true",
            help="Mass preservation",
        )

        parser.add_argument(
            "--scale-monotonic-energy",
            action="store_true",
            help="",
        )

        parser.add_argument(
            "--init-monotonic-bias",
            type=float,
            default=0.0,
            help="",
        )

        parser.add_argument(
            "--noise-var",
            type=float,
            default=1.0,
            help="",
        )

        parser.add_argument(
            "--noise-avg",
            type=float,
            default=0.0,
            help="",
        )

        parser.add_argument(
            "--monotonic-eps",
            type=float,
            default=1e-8,
            help="",
        )

    def set_target_step(self, step):
        self.target_step = step

    def get_target_step(self):
        return self.target_step

    def p_choose(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        decoder_state,
        encoder_out_dict,
        previous_attention=None,
        incremental_state=None,
        *args, **kargs
    ):
        """
        The expected input dimensions are:

        decoder_state: bsz x decoder_hidden_state_dim
        encoder_out: dict of encoder outputs
        previous_attention: src_len x bsz
        """

        if incremental_state is None:
            beta, alpha = self.attn_scores_train(
                decoder_state, encoder_out_dict, previous_attention)
        else:
            beta, alpha = self.attn_scores_infer(
                decoder_state, encoder_out_dict, incremental_state)

        # Sum weighted sources (bsz x context_dim)
        encoder_states = encoder_out_dict["encoder_out"]
        weighted_context = (
            encoder_states * beta.type_as(encoder_states).unsqueeze(2)
        ).sum(dim=0)

        #output = torch.tanh(self.output_proj(torch.cat([weighted_context, input], dim=1)))
        output = weighted_context

        return output, alpha

    def expected_hard_alignment(
        self, p_choose, previous_attention, encoder_padding_mask=None, eps=1e-8, *args
    ):
        """
        Calculate expected alignment given stepwise p_choose

        p_choose : src_len, bsz
        """
        src_len, bsz = p_choose.size()
        if previous_attention is None:
            previous_attention = p_choose.new_zeros([src_len, bsz])
            previous_attention[0] = 1
        else:
            previous_attention = previous_attention.squeeze(1)

        # src_len x bsz
        cumprod_one_minus_p_choose = exclusive_cumprod(1 - p_choose, dim=0, eps=self.eps)

        denominator = torch.clamp(cumprod_one_minus_p_choose, self.eps, 1.0)

        alpha = (
            p_choose
            * cumprod_one_minus_p_choose
            * torch.cumsum(
                previous_attention / denominator,
                dim=0,
            )
        )

        if self.mass_preservation:
            if encoder_padding_mask is not None:
                # Right side padding
                # pylint: disable=invalid-unary-operand-type
                eos_idx = (~encoder_padding_mask).sum(0, keepdim=True) - 1
                alpha.scatter_(0, eos_idx, 0)
                residual = 1.0 - torch.sum(alpha, dim=0, keepdim=True).clamp(0.0, 1.0)
                alpha.scatter_(0, eos_idx, residual)
            else:
                alpha[-1] = 1 - torch.sum(alpha[:-1], dim=0).clamp(0.0, 1.0)

        return alpha

    def expected_soft_alignment(
        self, alpha, decoder_hidden_state, encoder_hidden_state, encoder_padding_mask, *args
    ):
        raise NotImplementedError

    def attn_scores_train(
        self,
        decoder_state,
        encoder_out_dict,
        previous_attention=None,
    ):
        """The expected input dimensions are:
        input: bsz x decoder_hidden_state_dim
        encoder_padding_mask: src_len x bsz
        previous_attention: src_len x bsz
        """
        encoder_states = encoder_out_dict["encoder_out"]
        encoder_padding_mask = encoder_out_dict["encoder_padding_mask"]
        encoder_segmentation = encoder_out_dict.get("encoder_segmentation", None)
        # src_len x bsz
        p_choose = self.p_choose(
            decoder_state, encoder_states, encoder_padding_mask, encoder_segmentation
        )

        alpha = self.expected_hard_alignment(
            p_choose, previous_attention, encoder_padding_mask, encoder_segmentation
        )

        beta = self.expected_soft_alignment(
            alpha, decoder_state, encoder_states, encoder_padding_mask, encoder_segmentation
        )

        return beta, alpha

    def attn_scores_infer(
        self,
        decoder_state,
        encoder_out_dict,
        incremental_state,
    ):
        """The expected input dimensions are:
        input: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        monotonic_step: src_len x bsz
        """
        encoder_states = encoder_out_dict["encoder_out"]
        encoder_padding_mask = encoder_out_dict["encoder_padding_mask"]
        encoder_segmentation = encoder_out_dict.get("encoder_segmentation", None)
        max_src_len = encoder_states.size(0)
        # 1. Read previous steps
        prev_monotonic_step_list = utils.get_incremental_state(self, incremental_state, 'step')
        target_step = utils.get_incremental_state(self, incremental_state, 'target_step')
        if target_step is None:
            target_step = [0]
        self.set_target_step(target_step[-1])

        if prev_monotonic_step_list is None:
            prev_monotonic_step_list = []

        # 2. Update current steps
        curr_monotonic_step = self.update_monotonic_step(
            decoder_state,
            encoder_states,
            prev_monotonic_step_list,
            encoder_padding_mask,
            encoder_segmentation,
        )

        utils.set_incremental_state(
            self,
            incremental_state,
            "step",
            prev_monotonic_step_list + [curr_monotonic_step]
        )

        utils.set_incremental_state(
            self,
            incremental_state,
            "target_step",
            target_step + [target_step[-1] + 1]
        )
        # 3. Generate mask from steps
        monotonic_mask = self.monotonic_mask_from_step(curr_monotonic_step, max_src_len)
        # 4. Caculate energy
        exp_softattn_energy = self.softattn_energy_layer(
            decoder_state,
            encoder_states,
            encoder_padding_mask,
            exponential=True
        )
        masked_exp_softattn_energy = exp_softattn_energy * monotonic_mask.type_as(exp_softattn_energy)

        beta = masked_exp_softattn_energy / masked_exp_softattn_energy.sum(dim=0)

        alpha = (
            monotonic_mask
            ^ self.monotonic_mask_from_step(curr_monotonic_step, max_src_len)
        ).type_as(curr_monotonic_step)

        return beta, alpha

    def update_monotonic_step(
        self,
        decoder_state,
        source_hids,
        prev_monotonic_step_list,
        encoder_padding_mask,
        encoder_segmentation
    ):
        """The expected input dimensions are:
        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        prev_monotonic_step_list: list of 1 x bsz
        encoder_padding_mask: src_len x bsz
        """

        max_src_len, bsz, _ = source_hids.size()

        if len(prev_monotonic_step_list) == 0:
            prev_monotonic_step = (
                source_hids
                .new_zeros(1, bsz)
                .long()
            )
        else:
            prev_monotonic_step = prev_monotonic_step_list[-1].clone()

        # 1 x bsz
        if encoder_padding_mask is not None:
            src_lengths = (
                max_src_len
                - encoder_padding_mask
                .type_as(prev_monotonic_step)
                .sum(dim=0, keepdim=True)
            )
        else:
            src_lengths = max_src_len * torch.ones_like(prev_monotonic_step)

        # 1 x bsz
        new_monotonic_step = prev_monotonic_step
        # 1 x bsz
        finish_read = new_monotonic_step.eq(src_lengths - 1).long()

        while finish_read.sum().item() < bsz:
            # 1 x bsz
            p_choose = (
                self
                .p_choose(
                    decoder_state,
                    source_hids,
                    encoder_padding_mask,
                    encoder_segmentation,
                )
                .gather(0, new_monotonic_step)
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

    @staticmethod
    def monotonic_mask_from_step(
        monotonic_step,
        max_src_len
    ):
        """
        monotonic_step: 1, bsz
        max_src_len: scalar

        Convert a tensor of steps to mask
        For example, monotonic_step = [[2, 3, 4]], max_len = 5
        mask =
           [[1, 1, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]]
        """
        assert len(monotonic_step.size()) == 2
        batch_size = monotonic_step.size(1)

        # batch_size, max_len
        mask = (
            torch
            .arange(max_src_len)
            .unsqueeze(1)
            .expand(max_src_len, batch_size)
            .type_as(monotonic_step)
        ) < monotonic_step + 1

        return mask


@register_monotonic_attention("monotonic_hard")
class MonotonicHardAttentionLayer(MonotonicAttentionLayer):

    def add_discrete_noise(self, energy):
        if self.training:
            noise = (
                torch.normal(self.noise_avg, self.noise_var, energy.size())
                .type_as(energy)
                .to(energy.device)
            )
            return energy + noise
        else:
            return energy

    def p_choose(
        self,
        decoder_state,
        encoder_states,
        encoder_padding_mask,
        *args
    ):
        """The expected input dimensions are:
        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        """
        energy = self.monotonic_energy_layer(decoder_state, encoder_states, encoder_padding_mask)
        # src_len x bsz x 1
        p_choose = torch.sigmoid(self.add_discrete_noise(energy))

        return p_choose

    def expected_soft_alignment(self, alpha, *args):
        return alpha
