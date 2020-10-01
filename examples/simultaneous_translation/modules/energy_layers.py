import torch
from torch import nn
class EnergyLayer(nn.Module):
    """
    Module to calculate unormalized enegries.
    The energy then can be used to calculate softmax or monotonic attention.
    """
    def __init__(self, input_dim, context_dim, attention_dim=None, scale=False, init_bias=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        if attention_dim is None:
            self.attention_dim = self.input_dim
        else:
            self.attention_dim = attention_dim

        self.scale = scale
        self.init_bias = init_bias
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
            self.r = nn.Parameter(torch.ones(1) * self.init_bias)
        else:
            self.to_scores = nn.Linear(self.attention_dim, 1, bias=False)
            self.r = self.init_bias

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
        energy = self.to_scores(hidden_att).view(src_len, bsz) + self.r

        # Mask + softmax (src_len x bsz)
        if encoder_padding_mask is not None:
            energy = (
                energy.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(energy)
            )  # FP16 support: cast to float and back

        return energy