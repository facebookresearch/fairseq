# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from fairseq import utils

from fairseq.modules import MultiheadAttention

from examples.simultaneous_translation.utils.functions import (
    exclusive_cumprod,
    lengths_to_mask
)


from . import register_monotonic_attention


@register_monotonic_attention("hard_aligned")
class MonotonicMultiheadAttention(MultiheadAttention):

    def __init__(self, args):
        super().__init__(
            embed_dim=args.decoder_embed_dim,
            num_heads=args.decoder_attention_heads,
            kdim=getattr(args, 'encoder_embed_dim', None),
            vdim=getattr(args, 'encoder_embed_dim', None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True
        )
        self.bias = True
        self.eps = args.attention_eps
        self.mass_preservation = args.mass_preservation

        self.noise_type = args.noise_type
        self.noise_mean = args.noise_mean
        self.noise_var = args.noise_var

        self.energy_bias_init = args.energy_bias_init
        self.energy_bias = (
            nn.Parameter(self.energy_bias_init * torch.ones([1]))
            if args.energy_bias == True else 0
        )

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--mass-preservation', action="store_true", default=False,
                            help='Stay on the last token when decoding')
        parser.add_argument('--noise-var', type=float, default=1.0,
                            help='Variance of discretness noise')
        parser.add_argument('--noise-mean', type=float, default=0.0,
                            help='Mean of discretness noise')
        parser.add_argument('--noise-type', type=str, default="flat",
                            help='Type of discretness noise')
        parser.add_argument('--energy-bias', action="store_true", default=False,
                            help='')
        parser.add_argument('--energy-bias-init', type=float, default=-2.0,
                            help='')
        parser.add_argument('--reuse-attention', action='store_true', default=False)

        parser.add_argument('--attention-eps', type=float, default=1e-6)

    def forward(self, query, key, value, 
                incremental_state=None, key_padding_mask=None,
                need_weights=True, static_kv=False, attn_mask=None, 
                monotonic_step=None, encoder_decoder_attn=None, *args, **kwargs):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        tgt_len, bsz, embed_dim = query.size()

        # prepare inputs
        (
            q_proj, k_proj, v_proj,
            key_padding_mask, attn_mask 
        ) = self.multihead_inputs(
            query, key, value,
            incremental_state, key_padding_mask, attn_mask,
        )

        src_len = v_proj.size(1)

        if encoder_decoder_attn is not None:
            attn_weights = (
                encoder_decoder_attn[0]
                .view(bsz * self.num_heads, tgt_len, -1)
            )
        else:
            # stepwise prob
            # p_choose: bsz * self.num_heads, tgt_len, src_len
            p_choose = self.p_choose(q_proj, k_proj, key_padding_mask, attn_mask)
            if incremental_state is not None:
                alpha = self.expected_alignment_infer(p_choose, incremental_state, key_padding_mask)
            else:
                alpha = self.expected_alignment_train(p_choose, src_len)
            
            beta = self.expected_attention(
                alpha, query, key, value,
                incremental_state, key_padding_mask, attn_mask
            )

            attn_weights = beta

        attn = torch.bmm(attn_weights.type_as(v_proj), v_proj)

        attn = (
            attn
            .transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, embed_dim)
        )

        attn = self.out_proj(attn)

        beta = beta.view(bsz, self.num_heads, tgt_len, src_len)
        alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)

        return attn, {"alpha": alpha, "beta": beta, "p_choose": p_choose}

    def expected_attention(self, alpha, *args):
        return alpha

    def expected_alignment_train(self, p_choose, key_padding_mask):
        """
        Calculating expected alignment for MMA
        Mask is not need because p_choose will be 0 if masked

        q_ij = (1 − p_{ij−1})q_{ij−1} + a+{i−1j}
        a_ij = p_ij q_ij

        parellel solution:
        ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))

        ============================================================
        Expected input size
        p_choose: bsz * num_heads, tgt_len, src_len
        """

        # p_choose: bsz * num_heads, tgt_len, src_len
        bsz_num_heads, tgt_len, src_len = p_choose.size()

        # cumprod_1mp : bsz * num_heads, tgt_len, src_len
        cumprod_1mp = exclusive_cumprod(1 - p_choose, dim=2, eps=self.eps)
        cumprod_1mp_clamp = torch.clamp(cumprod_1mp, self.eps, 1.0)

        init_attention = p_choose.new_zeros([bsz_num_heads, 1, src_len])
        init_attention[:, :, 0] = 1.0

        previous_attn = [init_attention]

        for i in range(tgt_len):
            # p_choose: bsz * num_heads, tgt_len, src_len
            # cumprod_1mp_clamp : bsz * num_heads, tgt_len, src_len
            # previous_attn[i]: bsz * num_heads, 1, src_len
            # alpha_i: bsz * num_heads, src_len
            alpha_i = (
                p_choose[:, i]
                * cumprod_1mp[:, i]
                * torch.cumsum(
                    previous_attn[i][:, 0] / cumprod_1mp_clamp[:, i],
                    dim=1
                )
            )
            previous_attn.append(alpha_i.unsqueeze(1))

        # alpha: bsz * num_heads, tgt_len, src_len
        alpha = torch.cat(previous_attn[1:], dim=1)

        if self.mass_preservation:
            # Last token has the residual probabilities
            alpha[:, :, -1] = 1 - alpha[:, :, :-1].sum(dim=-1).clamp(0.0, 1.0)

        if torch.isnan(alpha).any():
            # Something is wrong
            raise RuntimeError("NaN in alpha.")

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def expected_alignment_infer(
        self, p_choose, incremental_state, encoder_padding_mask
    ):
        """
        Calculating monotonic alignment for MMA

        ============================================================
        Expected input size
        p_choose: bsz * num_heads, tgt_len, src_len
        incremental_state: dict
        encodencoder_padding_mask: bsz * src_len
        """
        # p_choose: bsz * self.num_heads, src_len
        bsz_num_heads, tgt_len, src_len = p_choose.size()
        # One token at a time
        assert tgt_len == 1
        p_choose = p_choose[:, 0, :]

        monotonic_cache = self._get_monotonic_buffer(incremental_state)

        # prev_monotonic_step: bsz, num_heads
        bsz = bsz_num_heads // self.num_heads
        prev_monotonic_step = monotonic_cache.get(
            "step",
            p_choose.new_zeros([bsz, self.num_heads]).long()
        )
        bsz, num_heads = prev_monotonic_step.size()
        assert num_heads == self.num_heads
        assert bsz * num_heads == bsz_num_heads

        # p_choose: bsz, num_heads, src_len
        p_choose = p_choose.view(bsz, num_heads, src_len)

        if encoder_padding_mask is not None:
            src_lengths = src_len - encoder_padding_mask.sum(dim=1, keepdim=True).long()
        else:
            src_lengths = prev_monotonic_step.new_ones(bsz, 1) * src_len

        # src_lengths: bsz, num_heads
        src_lengths = src_lengths.expand_as(prev_monotonic_step)
        # new_monotonic_step: bsz, num_heads
        new_monotonic_step = prev_monotonic_step

        step_offset = 0
        if encoder_padding_mask is not None:
            if encoder_padding_mask[:, 0].any():
                # left_pad_source = True:
                step_offset = encoder_padding_mask.sum(dim=-1, keepdim=True)

        max_steps = (
            src_lengths - 1 if self.mass_preservation
            else src_lengths
        )

        # finish_read: bsz, num_heads
        finish_read = new_monotonic_step.eq(max_steps)

        while finish_read.sum().item() < bsz * self.num_heads:
            # p_choose: bsz * self.num_heads, src_len
            # only choose the p at monotonic steps
            # p_choose_i: bsz , self.num_heads
            p_choose_i = (
                p_choose
                .gather(
                    2,
                    (step_offset + new_monotonic_step).unsqueeze(2)
                    .clamp(0, src_len - 1)
                )
            ).squeeze(2)

            action = (
                (p_choose_i < 0.5)
                .type_as(prev_monotonic_step)
                .masked_fill(finish_read, 0)
            )
            # 1 x bsz
            # sample actions on unfinished seq
            # 1 means stay, finish reading
            # 0 means leave, continue reading
            # dist = torch.distributions.bernoulli.Bernoulli(p_choose)
            # action = dist.sample().type_as(finish_read) * (1 - finish_read)

            new_monotonic_step += action

            finish_read = new_monotonic_step.eq(max_steps) | (action == 0)

        monotonic_cache["step"] = new_monotonic_step

        # alpha: bsz * num_heads, 1, src_len
        # new_monotonic_step: bsz, num_heads
        alpha = (
            p_choose
            .new_zeros([bsz * self.num_heads, src_len])
            .scatter(
                1,
                (step_offset + new_monotonic_step).view(bsz * self.num_heads, 1).clamp(0, src_len - 1),
                1
            )
        )

        if not self.mass_preservation:
            alpha = alpha.masked_fill(
                (new_monotonic_step == max_steps).view(bsz * self.num_heads, 1),
                0
            )

        alpha = alpha.unsqueeze(1)

        self._set_monotonic_buffer(incremental_state, monotonic_cache)

        return alpha

    def p_choose(self, q_proj, k_proj, key_padding_mask=None, attn_mask=None):
        """
        Calculating step wise prob for reading and writing
        1 to read, 0 to write

        ============================================================
        Expected input size
        query: bsz, tgt_len, embed_dim
        key: bsz, src_len, embed_dim
        value: bsz, src_len, embed_dim
        key_padding_mask: bsz, src_len
        attn_mask: bsz, src_len
        query: bsz, tgt_len, embed_dim
        """

        # attention energy
        attn_energy = self.multihead_energy(q_proj, k_proj, key_padding_mask, attn_mask)
        noise = 0

        if self.training:
            if self.noise_type == "flat":
                # add noise here to encourage discretness
                noise = (
                    torch
                    .normal(
                        self.noise_mean,
                        self.noise_var,
                        attn_energy.size()
                    )
                    .type_as(attn_energy)
                    .to(attn_energy.device)
                )

        p_choose = torch.sigmoid(attn_energy + noise)

        # p_choose: bsz * self.num_heads, tgt_len, src_len
        return p_choose

    def multihead_inputs(self, query, key, value, incremental_state=None, key_padding_mask=None,
                         attn_mask=None, static_kv=True, softmax=False):
        """
        Prepare inputs for multihead attention

        ============================================================
        Expected input size
        energy_type: monotonic or soft
        query: tgt_len, bsz, embed_dim
        key: src_len, bsz, embed_dim
        """
        prev_key_name = 'prev_key_soft' if softmax else 'prev_key'
        prev_value_name = 'prev_value_soft' if softmax else 'prev_value'

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        if softmax:
            q = self.in_proj_q_soft(query)
        else:
            q = self.q_proj(query)

        if key is None:
            # using prev_key and prev_value
            # assert value is None
            k = v = None
        else:
            if softmax:
                k = self.in_proj_k_soft(key)
            else:
                k = self.k_proj(key)
            if softmax:
                v = k
            else:
                v = self.v_proj(key)

        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if False and saved_state is not None:
            # TODO: Leave it for now because we are not doing incremental encoding
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if prev_key_name in saved_state:
                prev_key = saved_state[prev_key_name].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if prev_value_name in saved_state:
                prev_value = saved_state[prev_value_name].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state[prev_key_name] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state[prev_value_name] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        return q, k, v, key_padding_mask, attn_mask

    def multihead_energy(self, q_proj, k_proj, key_padding_mask=None, attn_mask=None):
        """
        Calculating multihead energies

        ============================================================
        Expected input size
        q_proj: bsz * num_heads, tgt_len, self.head_dim
        k_proj: bsz * num_heads, src_len, self.head_dim
        key_padding_mask: bsz, src_len
        attn_mask: tgt_len, src_len
        """
        bsz, tgt_len, embed_dim = q_proj.size()
        bsz = bsz // self.num_heads
        src_len = k_proj.size(1)

        attn_energy = torch.bmm(q_proj, k_proj.transpose(1, 2)) + self.energy_bias
        attn_energy = MultiheadAttention.apply_sparse_mask(attn_energy, tgt_len, src_len, bsz)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_energy.size(0), 1, 1)
            attn_energy += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_energy = attn_energy.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_energy = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_energy.float()
                ).type_as(attn_energy)
            else:
                attn_energy = attn_energy.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )

            attn_energy = attn_energy.view(bsz * self.num_heads, tgt_len, src_len)

        return attn_energy

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        super().reorder_incremental_state(incremental_state, new_order)
        input_buffer = self._get_monotonic_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_monotonic_buffer(incremental_state, input_buffer)

    def _get_monotonic_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'monotonic',
        ) or {}

    def _set_monotonic_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'monotonic',
            buffer,
        )
    
    def get_pointer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'monotonic',
        ) or {}
    
    def get_fastest_pointer(self, incremental_state):
        return self.get_pointer(incremental_state)["step"].max(0)[0]

    def set_pointer(self, incremental_state, p_choose):
        curr_pointer = self.get_pointer(incremental_state)
        if len(curr_pointer) == 0:
            buffer = torch.zeros_like(p_choose)
        else:
            buffer = self.get_pointer(incremental_state)["step"]

        
        #print(buffer.size(), p_choose.size())
        buffer += (p_choose < 0.5).type_as(buffer)
        
        utils.set_incremental_state(
            self,
            incremental_state,
            'monotonic',
            {"step": buffer},
        )


@register_monotonic_attention("infinite_lookback")
class MonotonicInfiniteLookbackMultiheadAttention(MonotonicMultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_soft_attention()

    def expected_attention(self, alpha, query, key, value, incremental_state, key_padding_mask, attn_mask):
        # monotonic attention, we will calculate milk here
        bsz_x_num_heads, tgt_len, src_len = alpha.size()
        bsz = int(bsz_x_num_heads / self.num_heads)

        q, k, v, key_padding_mask, attn_mask = self.multihead_inputs(
            query, key, value, incremental_state, key_padding_mask, attn_mask, softmax=True
        )
        soft_energy = torch.bmm(q, k.transpose(1, 2))
        #soft_energy = self.apply_sparse_mask(soft_energy, tgt_len, src_len, bsz)

        try:
            assert list(soft_energy.size()) == [bsz * self.num_heads, tgt_len, src_len]
        except:
            import pdb;pdb.set_trace()

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(soft_energy.size(0), 1, 1)
            soft_energy += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            soft_energy = soft_energy.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                soft_energy = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    soft_energy.float()
                ).type_as(soft_energy)
            else:
                soft_energy = soft_energy.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )
            soft_energy = soft_energy.view(bsz * self.num_heads, tgt_len, src_len)

        if incremental_state is not None:
            monotonic_cache = self._get_monotonic_buffer(incremental_state)
            monotonic_step = monotonic_cache["step"] + 1
            step_offset = 0
            if key_padding_mask is not None:
                if key_padding_mask[:, 0].any():
                    # left_pad_source = True:
                    step_offset = key_padding_mask.sum(dim=-1, keepdim=True)
            monotonic_step += step_offset
            mask = lengths_to_mask(monotonic_step.view(-1), soft_energy.size(2), 1).unsqueeze(1)
            
            soft_energy = soft_energy.masked_fill(~ mask, float('-inf'))
            soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
            exp_soft_energy = torch.exp(soft_energy)
            exp_soft_energy_sum = exp_soft_energy.sum(dim=2)
            beta = exp_soft_energy / exp_soft_energy_sum.unsqueeze(2)
            
        else:
            soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
            exp_soft_energy = torch.exp(soft_energy) + self.eps
            inner_items = alpha / (torch.cumsum(exp_soft_energy, dim=2))
            beta = exp_soft_energy * torch.cumsum(inner_items.flip(dims=[2]), dim=2).flip(dims=[2])
            beta = beta.view(bsz, self.num_heads, tgt_len, src_len).masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), 0)
            beta = beta / beta.sum(dim=3, keepdim=True)
            beta = beta.view(bsz * self.num_heads, tgt_len, src_len)
            beta = F.dropout(beta, p=self.dropout, training=self.training)

        if torch.isnan(beta).any():
            print(beta)
            import pdb;pdb.set_trace()

        return beta

    def init_soft_attention(self):
        embed_dim = self.embed_dim

        if self.qkv_same_dim:
            self.in_proj_weight_soft = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight_soft = Parameter(torch.Tensor(embed_dim, self.kdim))
            # self.v_proj_weight_soft = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight_soft = Parameter(torch.Tensor(embed_dim, embed_dim))

        if self.bias:
            self.in_proj_bias_soft = Parameter(torch.Tensor(2 * embed_dim))
        else:
            self.register_parameter('in_proj_bias_soft', None)

        #self.out_proj_soft = nn.Linear(embed_dim, embed_dim, bias=bias)

        if self.bias_k is not None and self.bias_v is not None:
            self.bias_k_soft = Parameter(torch.Tensor(1, 1, embed_dim))
            # self.bias_v_soft = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k_soft = self.bias_v_soft = None
        
        self.reset_parameters_soft()

    def reset_parameters_soft(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight_soft)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight_soft)
            # nn.init.xavier_uniform_(self.v_proj_weight_soft)
            nn.init.xavier_uniform_(self.q_proj_weight_soft)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias_soft is not None:
            nn.init.constant_(self.in_proj_bias_soft, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k_soft is not None:
            nn.init.xavier_normal_(self.bias_k_soft)
        # if self.bias_v_soft is not None:
            # nn.init.xavier_normal_(self.bias_v_soft)

    def in_proj_q_soft(self, query):
        if self.qkv_same_dim:
            return self._in_proj_soft(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias_soft
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight_soft, bias)

    def in_proj_k_soft(self, key):
        if self.qkv_same_dim:
            return self._in_proj_soft(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight_soft
            bias = self.in_proj_bias_soft
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    # def in_proj_v_soft(self, value):
    #     if self.qkv_same_dim:
    #         return self._in_proj_soft(value, start=2 * self.embed_dim)
    #     else:
    #         weight = self.v_proj_weight_soft
    #         bias = self.in_proj_bias_soft
    #         if bias is not None:
    #             bias = bias[2 * self.embed_dim:]
    #         return F.linear(value, weight, bias)

    def _in_proj_soft(self, input, start=0, end=None):
        weight = self.in_proj_weight_soft
        bias = self.in_proj_bias_soft
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
