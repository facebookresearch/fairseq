# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import distributed_utils as dist_utils
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, gelu
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.fused_bias_gelu import (
    fused_bias_gelu,
    has_fused_bias_gelu,
    has_megatron_fused_kernels,
    load_megatron_fused_kernel,
)
from fairseq.modules.fused_bias_relu_squared import fused_bias_relu_squared
from fairseq.modules.linear import Linear
from fairseq.modules.moe import CMRLayer, MOELayer, Top1Gate, Top2Gate
from fairseq.modules.quant_noise import quant_noise
from fairseq.utils import relu_squared


def _linear(x, weight, bias=None):
    return F.linear(x, weight, bias)


def _ffn(
    x,
    fc1,
    activation_fn,
    activation_dropout_module,
    fc2,
    dropout_module,
    ffn_ln=None,
):
    x_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    if has_fused_bias_gelu and activation_fn == gelu:
        x = _linear(x, fc1.weight)
        x = fused_bias_gelu(x, fc1.bias)
    elif activation_fn == relu_squared:
        x = _linear(x, fc1.weight)
        x = fused_bias_relu_squared(x, fc1.bias)
    else:
        x = _linear(x, fc1.weight, fc1.bias)
        x = activation_fn(x)
    x = activation_dropout_module(x)
    if ffn_ln is not None:
        x = ffn_ln(x)
    x = _linear(x, fc2.weight, fc2.bias)
    x = x.view(x_shape)
    fc_result = x
    x = dropout_module(x)
    return x, fc_result


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network layer in the Transformer model
    """

    def __init__(
        self, cfg, embed_dim, ffn_dim, dropout_module=None, init_model_on_gpu=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.activation_fn = utils.get_activation_fn(
            activation=str(cfg.activation_fn)
            if cfg.activation_fn is not None
            else "relu"
        )
        activation_dropout_p = cfg.activation_dropout or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.fc1 = self.build_fc(
            self.embed_dim,
            ffn_dim,
            self.quant_noise,
            self.quant_noise_block_size,
            init_model_on_gpu=init_model_on_gpu,
        )
        self.fc2 = self.build_fc(
            ffn_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
            init_model_on_gpu=init_model_on_gpu,
        )
        self.dropout_module = (
            FairseqDropout(cfg.dropout, module_name=self.__class__.__name__)
            if not dropout_module
            else dropout_module
        )

    def build_fc(
        self, input_dim, output_dim, q_noise, qn_block_size, init_model_on_gpu
    ):
        return quant_noise(
            Linear(input_dim, output_dim, init_model_on_gpu=init_model_on_gpu),
            p=q_noise,
            block_size=qn_block_size,
        )

    def forward(self, x):
        return _ffn(
            x,
            fc1=self.fc1,
            activation_fn=self.activation_fn,
            activation_dropout_module=self.activation_dropout_module,
            fc2=self.fc2,
            dropout_module=self.dropout_module,
        )[0]


class TransformerEncoderLayerBase(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg, return_fc=False, is_moe_layer=False):
        super().__init__()
        self.cfg = cfg
        self.return_fc = return_fc
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        # completetly drop the expert output of some tokens and rely on the residual connection
        self.moe_fom = cfg.moe_fom
        self.normalize_before = cfg.encoder_normalize_before
        self.is_moe_layer = is_moe_layer
        self.prefix_token_positions = (
            [0] if cfg.encoder_langtok in ["src", "tgt"] else None
        )
        ffn_dim = cfg.encoder.ffn_embed_dim
        self.attn_ln = LayerNorm(self.embed_dim) if cfg.scale_attn else None
        self.ffn_layernorm = LayerNorm(ffn_dim) if cfg.scale_fc else None
        if self.is_moe_layer and cfg.alternate_ffn_embed_dim > 0:
            ffn_dim = cfg.alternate_ffn_embed_dim
        # the second condition is for a "pseudo" MoE layer
        # (shared FFN with expert FFN dimension) that tries
        # to replicate FLOPs used by an expert MoE layer with perfectly balanced load
        build_moe = self.is_moe_layer and cfg.alternate_ffn_embed_dim == 0
        if not build_moe or cfg.moe_cmr:
            self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
            activation_dropout_p = cfg.activation_dropout
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = cfg.relu_dropout or 0
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
            self.fc1 = self.build_fc1(
                self.embed_dim,
                ffn_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                ffn_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
        if build_moe:
            lang_idx = None
            if cfg.cmr_log_lang_gates:
                lang_idx = getattr(cfg, "lang_idx", None)
                assert lang_idx is not None, cfg
            if cfg.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    cfg.moe_expert_count,
                    use_fp32=cfg.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=cfg.moe_eval_capacity_token_fraction,
                    use_tutel=cfg.use_tutel_moe,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    cfg.moe_expert_count,
                    cfg.moe_gating_use_fp32,
                    cfg.moe_second_expert_policy,
                    cfg.moe_normalize_gate_prob_before_dropping,
                    cfg.moe_eval_capacity_token_fraction,
                    cfg.moe_batch_prioritized_routing,
                    use_tutel=cfg.use_tutel_moe,
                    init_model_on_gpu=cfg.init_model_on_gpu,
                )
            experts = make_experts(cfg, self.embed_dim, ffn_dim, self.dropout_module)
            self.moe_layer = MOELayer(
                gate,
                experts,
                cfg,
                max_positions=cfg.max_source_positions,
                tok_dropout=cfg.moe_eom,
                moe_local_drop=cfg.moe_local_drop,
            )
            if cfg.moe_cmr:
                self.cmr_layer = CMRLayer(
                    self.moe_layer,
                    lambda x: _ffn(
                        x,
                        self.fc1,
                        self.activation_fn,
                        self.activation_dropout_module,
                        self.fc2,
                        self.dropout_module,
                        ffn_ln=self.ffn_layernorm,
                    )[0],
                    self.embed_dim,
                    cfg.cmr_gate_drop,
                    lang_idx=lang_idx,
                )
        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def _get_fc_rank(self, remove_num: int) -> List[int]:
        f1_filter_param = []
        for i in range(self.fc1.out_features):
            f1_filter_param.append(
                torch.sum(torch.abs(self.fc1.weight[i]))
                + torch.sum(torch.abs(self.fc2.weight[:, i]))
                + torch.abs(self.fc1.bias[i])
            )
        return sorted(
            range(len(f1_filter_param)), key=lambda k: f1_filter_param[k], reverse=False
        )[0:remove_num]

    def _prune_fc_layer(self, remove_index: List[int]):
        new_fc1_weight = []
        new_fc1_bias = []
        for i in range(self.fc1.out_features):
            if i not in remove_index:
                new_fc1_weight.append(self.fc1.weight[i])
                new_fc1_bias.append(self.fc1.bias[i])

        new_fc1_weight = torch.stack(new_fc1_weight).detach()
        new_fc1_weight.requires_grad = True

        new_fc1_bias = torch.stack(new_fc1_bias).detach()
        new_fc1_bias.requires_grad = True

        self.fc1 = quant_noise(
            nn.Linear(self.fc1.in_features, self.fc1.out_features - len(remove_index)),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc1.weight = torch.nn.Parameter(new_fc1_weight)
        self.fc1.bias = torch.nn.Parameter(new_fc1_bias)

        new_fc2_weight = []
        new_fc2_bias = []
        for i in range(self.fc2.in_features):
            if i not in remove_index:
                new_fc2_weight.append(self.fc2.weight[:, i])
        new_fc2_bias = self.fc2.bias.detach()

        new_fc2_weight = torch.stack(new_fc2_weight, dim=-1).detach()
        new_fc2_weight.requires_grad = True

        new_fc2_bias = self.fc2.bias.detach()
        new_fc2_bias.requires_grad = True

        self.fc2 = quant_noise(
            nn.Linear(self.fc2.in_features - len(remove_index), self.fc2.out_features),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc2.weight = torch.nn.Parameter(new_fc2_weight)
        self.fc2.bias = torch.nn.Parameter(new_fc2_bias)

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            scale_heads=cfg.scale_heads,
            use_fused_softmax=cfg.use_fused_softmax,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        run_moe = self.is_moe_layer and self.cfg.alternate_ffn_embed_dim == 0
        if not run_moe:
            x, fc_result = _ffn(
                x,
                self.fc1,
                self.activation_fn,
                self.activation_dropout_module,
                self.fc2,
                self.dropout_module,
                ffn_ln=self.ffn_layernorm,
            )
            l_aux = None
        else:
            fc_result = None
            # x - seq_len, batch_size, model_dim
            x = x.transpose(0, 1)  # batch_size, seq_len, model_dim
            prefix_tokens = (
                tokens[:, self.prefix_token_positions]
                if tokens is not None and self.prefix_token_positions is not None
                else None
            )
            if self.cfg.moe_cmr:
                moe_module = self.cmr_layer
            else:
                moe_module = self.moe_layer
            if self.cfg.use_moe_pad_mask:
                x, l_aux = moe_module(
                    x,
                    input_padding_mask=encoder_padding_mask,
                    prefix_tokens=prefix_tokens,
                )
            else:
                x, l_aux = moe_module(x, prefix_tokens=prefix_tokens)

            if self.moe_fom:
                if self.training:
                    mask = (
                        torch.empty(x.shape[:-1], device=x.device).uniform_()
                        > self.moe_fom
                    )
                    x = mask.unsqueeze(-1) * x
                else:
                    x = (1 - self.moe_fom) * x

            x = x.transpose(0, 1)  # seq_len, batch_size, model_dim
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.return_fc and not torch.jit.is_scripting():
            return x, fc_result
        return x, l_aux


# backward compatible with the legacy argparse format
class TransformerEncoderLayer(TransformerEncoderLayerBase):
    def __init__(self, args, return_fc=False, is_moe_layer=False):
        from fairseq.models.transformer import TransformerConfig

        super().__init__(
            TransformerConfig.from_namespace(args),
            return_fc=return_fc,
            is_moe_layer=is_moe_layer,
        )
        self.args = args

    def build_self_attention(self, embed_dim, args):
        from fairseq.models.transformer import TransformerConfig

        return super().build_self_attention(
            embed_dim, TransformerConfig.from_namespace(args)
        )


class TransformerDecoderLayerBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
        is_moe_layer=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.decoder.embed_dim
        if has_megatron_fused_kernels and cfg.activation_fn == "gelu":
            load_megatron_fused_kernel()

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        # completetly drop the expert output of some tokens and rely on the residual connection
        self.moe_fom = cfg.moe_fom

        self.quant_noise = cfg.quant_noise_pq
        self.quant_noise_block_size = cfg.quant_noise_pq_block_size

        self.cross_self_attention = cfg.cross_self_attention
        self.attn_ln = LayerNorm(self.embed_dim) if cfg.scale_attn else None

        init_model_on_gpu = cfg.init_model_on_gpu
        if self.attn_ln is not None and init_model_on_gpu:
            self.attn_ln = self.attn_ln.cuda().half()

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim
        scale_heads = utils.safe_getattr(cfg, "scale_heads", False)
        init_tensor = torch.ones((self.nh,))
        if init_model_on_gpu:
            init_tensor = init_tensor.cuda().half()
        self.c_attn = (
            nn.Parameter(init_tensor, requires_grad=True) if scale_heads else None
        )

        # self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        # activation_dropout_p = cfg.activation_dropout
        # if activation_dropout_p == 0:
        #     # for backwards compatibility with models that use cfg.relu_dropout
        #     activation_dropout_p = cfg.relu_dropout or 0
        # self.activation_dropout_module = FairseqDropout(
        #     float(activation_dropout_p), module_name=self.__class__.__name__
        # )
        self.normalize_before = cfg.decoder.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if init_model_on_gpu:
            self.self_attn_layer_norm = self.self_attn_layer_norm.cuda().half()

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

            if init_model_on_gpu:
                self.encoder_attn_layer_norm = (
                    self.encoder_attn_layer_norm.cuda().half()
                )

        self.is_moe_layer = is_moe_layer
        self.prefix_token_positions = [1] if cfg.decoder_langtok else None

        ffn_dim = cfg.decoder.ffn_embed_dim
        if self.is_moe_layer and cfg.alternate_decoder_ffn_embed_dim > 0:
            ffn_dim = cfg.alternate_decoder_ffn_embed_dim

        self.alpha2 = None
        build_moe = self.is_moe_layer and cfg.alternate_decoder_ffn_embed_dim == 0
        if not build_moe or cfg.moe_cmr:
            self.activation_fn = utils.get_activation_fn(
                activation=str(cfg.activation_fn)
                if cfg.activation_fn is not None
                else "relu"
            )
            activation_dropout_p = cfg.activation_dropout or 0
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = cfg.relu_dropout or 0
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
            self.fc1 = self.build_fc1(
                self.embed_dim,
                ffn_dim,
                self.quant_noise,
                self.quant_noise_block_size,
                init_model_on_gpu=init_model_on_gpu,
            )
            self.ffn_layernorm = (
                LayerNorm(cfg.decoder.ffn_embed_dim)
                if utils.safe_getattr(cfg, "scale_fc", False)
                else None
            )
            if self.ffn_layernorm and init_model_on_gpu:
                self.ffn_layernorm = self.ffn_layernorm.cuda().half()

            if utils.safe_getattr(cfg, "scale_resids", False):
                self.alpha2 = nn.Parameter(
                    torch.ones(
                        self.embed_dim,
                        device=torch.cuda.current_device(),
                        dtype=torch.float16,
                    ),
                    requires_grad=True,
                )
            self.fc2 = self.build_fc2(
                ffn_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
                init_model_on_gpu=init_model_on_gpu,
            )
        if build_moe:
            lang_idx = None
            if cfg.cmr_log_lang_gates:
                lang_idx = getattr(cfg, "lang_idx")
                assert lang_idx is not None, cfg
            if cfg.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    cfg.moe_expert_count,
                    use_fp32=cfg.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=cfg.moe_eval_capacity_token_fraction,
                    use_tutel=cfg.use_tutel_moe,
                    init_model_on_gpu=init_model_on_gpu,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    cfg.moe_expert_count,
                    cfg.moe_gating_use_fp32,
                    cfg.moe_second_expert_policy,
                    cfg.moe_normalize_gate_prob_before_dropping,
                    cfg.moe_eval_capacity_token_fraction,
                    cfg.moe_batch_prioritized_routing,
                    use_tutel=cfg.use_tutel_moe,
                    init_model_on_gpu=init_model_on_gpu,
                )
            experts = make_experts(cfg, self.embed_dim, ffn_dim, self.dropout_module)
            self.moe_layer = MOELayer(
                gate,
                experts,
                cfg,
                max_positions=cfg.max_target_positions,
                tok_dropout=cfg.moe_eom,
                moe_local_drop=cfg.moe_local_drop,
            )
            if cfg.moe_cmr:
                self.cmr_layer = CMRLayer(
                    self.moe_layer,
                    lambda x: _ffn(
                        x,
                        self.fc1,
                        self.activation_fn,
                        self.activation_dropout_module,
                        self.fc2,
                        self.dropout_module,
                        ffn_ln=self.ffn_layernorm,
                    )[0],
                    self.embed_dim,
                    cfg.cmr_gate_drop,
                    lang_idx=lang_idx,
                )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        if init_model_on_gpu:
            self.final_layer_norm = self.final_layer_norm.cuda().half()
            for p in self.modules():
                p = p.cuda().half()
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(
        self,
        input_dim,
        output_dim,
        q_noise,
        qn_block_size,
        init_model_on_gpu=False,
    ):
        return quant_noise(
            Linear(input_dim, output_dim, init_model_on_gpu=init_model_on_gpu),
            q_noise,
            qn_block_size,
        )

    def build_fc2(
        self,
        input_dim,
        output_dim,
        q_noise,
        qn_block_size,
        init_model_on_gpu=False,
    ):
        return quant_noise(
            Linear(input_dim, output_dim, init_model_on_gpu=init_model_on_gpu),
            q_noise,
            qn_block_size,
        )

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            use_fused_softmax=cfg.use_fused_softmax,
            scale_heads=cfg.scale_heads_inside,
            init_model_on_gpu=cfg.init_model_on_gpu,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            use_fused_softmax=cfg.use_fused_softmax,
            init_model_on_gpu=cfg.init_model_on_gpu,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual, alpha=None):
        if alpha is None:
            return residual + x
        else:
            return x + torch.mul(alpha, residual)

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        tokens: Optional[Tensor] = None,
    ) -> Tuple[
        Tensor, Tensor, Optional[List[Optional[Tensor]]], Optional[Dict[str, Tensor]]
    ]:
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
            tokens (Tensor, optional): previous output tokens.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbdh", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer or self.cfg.alternate_decoder_ffn_embed_dim > 0:
            x, _ = _ffn(
                x,
                fc1=self.fc1,
                activation_fn=self.activation_fn,
                activation_dropout_module=self.activation_dropout_module,
                ffn_ln=self.ffn_layernorm,
                fc2=self.fc2,
                dropout_module=self.dropout_module,
            )
            l_aux = None
        else:
            # x - seq_len, batch_size, model_dim
            x = x.transpose(0, 1)  # batch_size, seq_len, model_dim
            prefix_tokens = (
                tokens[:, self.prefix_token_positions]
                if tokens is not None and self.prefix_token_positions is not None
                else None
            )
            if self.cfg.moe_cmr:
                moe_module = self.cmr_layer
            else:
                moe_module = self.moe_layer
            if self.cfg.use_moe_pad_mask:
                x, l_aux = moe_module(
                    x,
                    input_padding_mask=self_attn_padding_mask,
                    prefix_tokens=prefix_tokens,
                )
            else:
                x, l_aux = moe_module(x, prefix_tokens=prefix_tokens)

            if self.moe_fom:
                if self.training:
                    mask = (
                        torch.empty(x.shape[:-1], device=x.device).uniform_()
                        > self.moe_fom
                    )
                    x = mask.unsqueeze(-1) * x
                else:
                    x = (1 - self.moe_fom) * x

            x = x.transpose(0, 1)  # seq_len, batch_size, model_dim
        x = self.residual_connection(x, residual, alpha=self.alpha2)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None, l_aux

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


# backward compatible with the legacy argparse format
class TransformerDecoderLayer(TransformerDecoderLayerBase):
    def __init__(
        self,
        args,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
        is_moe_layer=False,
    ):
        from fairseq.models.transformer import TransformerConfig

        super().__init__(
            TransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            is_moe_layer=is_moe_layer,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        from fairseq.models.transformer import TransformerConfig

        return super().build_self_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        from fairseq.models.transformer import TransformerConfig

        return super().build_encoder_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
        )


def make_experts(cfg, embed_dim, expert_ffn_dim, dropout_module) -> nn.ModuleList:
    world_size = (
        1
        if not torch.distributed.is_initialized()
        else torch.distributed.get_world_size()
    )
    expert_list = []
    ddp_rank = dist_utils.get_data_parallel_rank()
    start_seed = torch.randint(1000000, (1,)).item()

    if cfg.moe_expert_count >= world_size:  # at least as many experts than gpus
        assert (
            cfg.moe_expert_count % world_size == 0
        ), f"{cfg.moe_expert_count}, {world_size}"
        local_moe_expert_count = cfg.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with utils.set_torch_seed(
                start_seed + ddp_rank * local_moe_expert_count + i
            ):
                expert_list.append(
                    FeedForwardNetwork(cfg, embed_dim, expert_ffn_dim, dropout_module)
                )

    else:  # less experts than gpus
        assert (
            world_size % cfg.moe_expert_count == 0
        ), f"{world_size}, {cfg.moe_expert_count}"
        # initialize each FFN with the same seed on different GPUs
        with utils.set_torch_seed(start_seed + ddp_rank % cfg.moe_expert_count):
            expert_list.append(
                FeedForwardNetwork(cfg, embed_dim, expert_ffn_dim, dropout_module)
            )
    experts = nn.ModuleList(expert_list)
    return experts
