import torch
from torch import nn
from fairseq import utils
from . import register_monotonic_attention
from .energy_layers import AdditiveEnergyLayer

@register_monotonic_attention("waitk")
class WaitKAttentionLayer(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.decoder_hidden_dim = args.decoder_hidden_dim
        self.encoder_hidden_dim = args.encoder_hidden_size

        self.softattn_energy_layer = AdditiveEnergyLayer(
            self.decoder_hidden_dim,
            self.encoder_hidden_dim,
        )

        self.waitk = args.waitk_lagging
        self.stride = args.waitk_stride

        self.pooling_layer = torch.nn.AvgPool1d(
           kernel_size=self.stride, 
           stride=self.stride,
           ceil_mode=True
        )

        self.waitk_stride = self.waitk * self.stride

        self.target_step = 0

        self.eps = 1e-8

    @staticmethod
    def add_args(parser):

        parser.add_argument(
            "--waitk-lagging",
            type=int,
            required=True,
            help="Lagging in Wait-K policy (K)",
        )

        parser.add_argument(
            "--waitk-stride",
            type=int,
            default=1,
            help="Size of fixed stride on source side",
        )

    def forward(
        self,
        input,
        source_hids,
        encoder_padding_mask,
        previous_attention=None,
        incremental_state=None,
        *args, **kargs
    ):
        """
        The expected input dimensions are:
        
        input: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        previous_attention: src_len x bsz
        """
        if incremental_state is None:
            beta, alpha = self.attn_scores_train(
                input, source_hids, encoder_padding_mask, previous_attention)
        else:
            beta, alpha = self.attn_scores_infer(
                input, source_hids, encoder_padding_mask, incremental_state)

        # Sum weighted sources (bsz x context_dim)
        weighted_context = (
            source_hids * beta.type_as(source_hids).unsqueeze(2)
        ).sum(dim=0)

        #output = torch.tanh(self.output_proj(torch.cat([weighted_context, input], dim=1)))
        output = weighted_context

        return output, alpha

    def attn_scores_train(
        self,
        decoder_state,
        encoder_states,
        encoder_padding_mask,
        previous_attention=None,
    ):
        """The expected input dimensions are:
        input: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        previous_attention: src_len x bsz
        """
        # src_len x bsz
        p_choose = self.p_choose(
            decoder_state, encoder_states, encoder_padding_mask
        )
        
        alpha = self.expected_hard_alignment(
            p_choose, previous_attention, encoder_padding_mask
        )

        beta = self.expected_soft_alignment(
            alpha, decoder_state, encoder_states, encoder_padding_mask
        )

        return beta, alpha

    def attn_scores_infer(
        self,
        input,
        source_hids,
        encoder_padding_mask,
        incremental_state,
    ):
        """The expected input dimensions are:
        input: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        monotonic_step: src_len x bsz
        """
        max_src_len = source_hids.size(0)
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
            input, 
            source_hids, 
            prev_monotonic_step_list, 
            encoder_padding_mask
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
            input, source_hids, encoder_padding_mask, exponential=True
        )
        masked_exp_softattn_energy = exp_softattn_energy * monotonic_mask.type_as(exp_softattn_energy)

        beta = masked_exp_softattn_energy / masked_exp_softattn_energy.sum(dim=0)

        alpha = (
            monotonic_mask 
            ^ self.monotonic_mask_from_step(curr_monotonic_step - 1, max_src_len)
        ).type_as(curr_monotonic_step)

        return beta, alpha

    def update_monotonic_step(
        self,
        decoder_state,
        source_hids,
        prev_monotonic_step_list,
        encoder_padding_mask
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
            prev_monotonic_step = prev_monotonic_step_list[-1]

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
        new_monotonic_step = prev_monotonic_step.clone()
        # 1 x bsz
        finish_read = new_monotonic_step.eq(src_lengths - 1).long()

        while finish_read.sum().item() < bsz:
            # 1 x bsz
            p_choose = (
                self
                .p_choose(decoder_state, source_hids)
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

    def p_choose(self,
        decoder_state,
        encoder_states,
        encoder_padding_mask=None
    ):
        """The expected input dimensions are:
        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        """
        src_len, bsz, _ = encoder_states.size()
        p_choose = decoder_state.new_zeros(src_len, bsz)
        p_choose[self.get_encoder_pointer(src_len) - 1:] = 1
        return p_choose

    def expected_hard_alignment(
        self, p_choose, previous_attention, encoder_padding_mask=None, eps=1e-8
    ):
        # src_len x bsz
        src_len, bsz = p_choose.size()
        alpha = p_choose.new_zeros(src_len, bsz)
        alpha[self.get_encoder_pointer(src_len) - 1] = 1.0
        return alpha

    def expected_soft_alignment(
        self, alpha, decoder_state, encoder_states, encoder_padding_mask
    ):
        src_len, bsz, _ = encoder_states.size()
        softattn_energy = self.softattn_energy_layer(
            decoder_state, encoder_states, encoder_padding_mask, exponential=False
        ) 
        softattn_energy_max, _ = torch.max(softattn_energy, dim=0)
        exp_softattn_energy = torch.exp(softattn_energy - softattn_energy_max) + self.eps
        exp_softattn_energy[self.get_encoder_pointer(src_len):] = 0
        
        beta = exp_softattn_energy / exp_softattn_energy.sum(dim=0, keepdim=True)
           
        return beta

    def get_pointer(self, src_len):
        pointer = self.get_target_step() + self.waitk - 1
        pointer_stride = min((pointer + 1) * self.stride, src_len - 1)
        return pointer, pointer_stride
    
    def get_encoder_pointer(self, src_len):
        return self.get_pointer(src_len)[-1]

    def set_target_step(self, step):
        self.target_step = step

    def get_target_step(self):
        return self.target_step

