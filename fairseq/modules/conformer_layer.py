# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from typing import Optional
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    ESPNETMultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
)
from fairseq.utils import get_activation_fn


class ConvolutionModule(torch.nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(
        self,
        embed_dim,
        channels,
        depthwise_kernel_size,
        dropout,
        activation_fn="swish",
        bias=False,
        export=False,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            channels: Number of channels in depthwise conv layers
            depthwise_kernel_size: Depthwise conv layer kernel size
            dropout: dropout value
            activation_fn: Activation function to use after depthwise convolution kernel
            bias: If bias should be added to conv layers
            export: If layernorm should be exported to jit
        """
        super(ConvolutionModule, self).__init__()
        assert (
            depthwise_kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.layer_norm = LayerNorm(embed_dim, export=export)
        self.pointwise_conv1 = torch.nn.Conv1d(
            embed_dim,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.batch_norm = torch.nn.BatchNorm1d(channels)
        self.activation = get_activation_fn(activation_fn)(channels)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        x = self.layer_norm(x)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in conformer"""

    def __init__(
        self,
        input_feat,
        hidden_units,
        dropout1,
        dropout2,
        activation_fn="swish",
        bias=True,
    ):
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            dropout1: dropout value for layer1
            dropout2: dropout value for layer2
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """

        super(FeedForwardModule, self).__init__()
        self.layer_norm = LayerNorm(input_feat)
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = torch.nn.Dropout(dropout1)
        self.dropout2 = torch.nn.Dropout(dropout2)
        self.activation = get_activation_fn(activation_fn)(hidden_units)

    def forward(self, x):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)


class ConformerEncoderLayer(torch.nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100. We currently don't support relative positional encoding in MHA"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout,
        use_fp16,
        depthwise_conv_kernel_size=31,
        activation_fn="swish",
        attn_type=None,
        pos_enc_type="abs",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            ffn_embed_dim: FFN layer dimension
            attention_heads: Number of attention heads in MHA
            dropout: dropout value
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
            activation_fn: Activation function name to use in convulation block and feed forward block
            attn_type: MHA implementation from ESPNET vs fairseq
            pos_enc_type: Positional encoding type - abs, rope, rel_pos
        """
        self.pos_enc_type = pos_enc_type
        super(ConformerEncoderLayer, self).__init__()

        self.ffn1 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
        )

        self.self_attn_layer_norm = LayerNorm(embed_dim, export=False)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos":
                self.self_attn = RelPositionMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            elif self.pos_enc_type == "rope":
                self.self_attn = RotaryPositionMultiHeadedAttention(
                    embed_dim, attention_heads, dropout=dropout, precision=use_fp16
                )
            elif self.pos_enc_type == "abs":
                self.self_attn = ESPNETMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            else:
                raise Exception(f"Unsupported attention type {self.pos_enc_type}")
        else:
            # Default to fairseq MHA
            self.self_attn = MultiheadAttention(
                embed_dim,
                attention_heads,
                dropout=dropout,
            )

        self.conv_module = ConvolutionModule(
            embed_dim=embed_dim,
            channels=embed_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation_fn=activation_fn,
        )

        self.ffn2 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
            activation_fn=activation_fn,
        )
        self.final_layer_norm = LayerNorm(embed_dim, export=False)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
        position_emb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        if self.pos_enc_type == "rel_pos":
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                pos_emb=position_emb,
                need_weights=False,
            )
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        # TBC to BTC
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        # BTC to TBC
        x = x.transpose(0, 1)
        x = residual + x

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x, attn


class ConformerWav2Vec2EncoderLayer(ConformerEncoderLayer):
    """Encoder layer for Wav2vec2 encoder"""

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        position_emb=None,
    ):
        return super().forward(x, self_attn_padding_mask, position_emb)
