import torch
from torch import nn
from fairseq import utils
from . import register_monotonic_attention
from .energy_layers import AdditiveEnergyLayer
READ=0
WRITE=1


@register_monotonic_attention("waitk_rnn")
class RNNWaitKAttentionLayer(nn.Module):

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

        self.waitk_stride = self.waitk * self.stride

        self.target_step = 0

        self.eps = 1e-8

        self.online_decode = True

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
        attn = self.attn_scores(input, source_hids, encoder_padding_mask, incremental_state)

        # Sum weighted sources (bsz x context_dim)
        weighted_context = (
            source_hids * attn.type_as(source_hids).unsqueeze(2)
        ).sum(dim=0)

        #output = torch.tanh(self.output_proj(torch.cat([weighted_context, input], dim=1)))
        output = weighted_context

        return output

    def attn_scores(
        self,
        decoder_state,
        encoder_states,
        encoder_padding_mask,
        incremental_state
    ):
        src_len, bsz, _ = encoder_states.size()
        softattn_energy = self.softattn_energy_layer(
            decoder_state, encoder_states, encoder_padding_mask, exponential=False
        )
        softattn_energy_max, _ = torch.max(softattn_energy, dim=0)
        exp_softattn_energy = torch.exp(softattn_energy - softattn_energy_max) + self.eps

        if incremental_state is None:
            exp_softattn_energy[self.get_encoder_pointer(src_len) + 1:] = 0

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

    def decision_from_states(self, states, frame_shift=1, subsampling_factor=1):
        if len(states["indices"]["src"]) == 0:
            # This mean that the utterence is too small
            return READ

        lagging = (
            (states["steps"]["tgt"] + self.waitk)
            * self.segment_size(frame_shift, subsampling_factor)
        )
        if lagging - states["steps"]["src"] > 0:
            return READ
        else:
            return WRITE

    def segment_size(self, frame_shift=1, subsampling_factor=1):
        return self.stride * subsampling_factor * frame_shift
