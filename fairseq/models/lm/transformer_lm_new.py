# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist

from dataclasses import dataclass, field
from omegaconf import II


from fairseq.dataclass import ChoiceEnum, FairseqDataclass

from fairseq import options, utils, checkpoint_utils, tasks
from fairseq.distributed import fsdp_wrap
from fairseq.models import BaseFairseqModel, FairseqIncrementalDecoder, register_model

from fairseq.models.transformer import (
    TransformerConfig,
    DEFAULT_MIN_PARAMS_TO_WRAP,
    Embedding,
    TransformerDecoderBase,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    AdaptiveInput,
    cross_entropy,
    EMAModule,
    EMAModuleConfig,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from . import transformer_layer

logger = logging.getLogger(__name__)


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@dataclass
class TransformerLMConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    output_dim: int = field(default=512, metadata={"help": "decoder output dimension"})
    input_dim: int = field(default=512, metadata={"help": "decoder input dimension"})
    ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    layers: int = field(default=6, metadata={"help": "num decoder layers"})
    attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    character_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, uses character embedding convolutions to produce token embeddings"
        },
    )
    character_filters: str = field(
        default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
        metadata={"help": "size of character embeddings"},
    )
    character_embedding_dim: int = field(
        default=4, metadata={"help": "size of character embeddings"}
    )
    char_embedder_highway_layers: int = field(
        default=2,
        metadata={"help": "number of highway layers for character token embeddder"},
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_input_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    adaptive_input_cutoff: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of adaptive input cutoff points."},
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        },
    )
    # config for "BASE Layers: Simplifying Training of Large, Sparse Models"
    base_layers: Optional[int] = field(
        default=0, metadata={"help": "number of BASE layers in total"}
    )
    base_sublayers: Optional[int] = field(
        default=1, metadata={"help": "number of sublayers in each BASE layer"}
    )
    base_shuffle: Optional[int] = field(
        default=1,
        metadata={"help": "shuffle tokens between workers before computing assignment"},
    )
    # NormFormer
    scale_fc: Optional[bool] = field(
        default=False,
        metadata={"help": "Insert LayerNorm between fully connected layers"},
    )
    scale_attn: Optional[bool] = field(
        default=False, metadata={"help": "Insert LayerNorm after attention"}
    )
    scale_heads: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each attention head"},
    )
    scale_resids: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each residual connection"},
    )
    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")

    detach_lm: bool = False
    use_fc_target: bool = True
    average_top_k_layers: int = field(
        default=0, metadata={"help": "how many layers to average"}
    )
    num_forward: int = 1
    avg_num_forward: bool = False
    recon_depth: int = 1

    layer_norm_target_layer: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    pretrained_target_model_path: Optional[str] = None
    predict_teacher_targets: bool = True

    alt_resid: bool = False


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


@register_model("transformer_lm_new", dataclass=TransformerLMConfig)
class TransformerLM(FairseqIncrementalDecoder, BaseFairseqModel):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg: TransformerLMConfig,
        dictionary,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self._future_mask = torch.empty(0)

        self.cross_self_attention = False

        if cfg.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(dictionary),
                dictionary.pad(),
                cfg.input_dim,
                cfg.adaptive_input_factor,
                cfg.embed_dim,
                options.eval_str_list(cfg.adaptive_input_cutoff, type=int),
                cfg.quant_noise_pq,
                cfg.quant_noise_pq_block_size,
            )

            if cfg.tie_adaptive_weights:
                assert cfg.adaptive_input_factor == cfg.adaptive_softmax_factor
                assert (
                    cfg.adaptive_softmax_cutoff == cfg.adaptive_input_cutoff
                ), "{} != {}".format(
                    cfg.adaptive_softmax_cutoff, cfg.adaptive_input_cutoff
                )
                assert cfg.input_dim == cfg.output_dim

        else:
            embed_tokens = Embedding(len(dictionary), cfg.input_dim, dictionary.pad())

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.layerdrop = cfg.layerdrop
        self.share_input_output_embed = cfg.share_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions or cfg.tokens_per_sample

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise_pq,
                cfg.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([self.build_layer(cfg) for _ in range(cfg.layers)])
        self.num_layers = len(self.layers)

        if cfg.normalize_before and not cfg.no_final_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None

        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

        self.num_updates = 0
        self.ema = None
        self.recon_proj = None
        self.target_model = None

        if cfg.average_top_k_layers > 0:
            def make_proj(in_dim, out_dim, depth):
                projs = []
                for _ in range(depth - 1):
                    projs.append(nn.Linear(in_dim, in_dim))
                    projs.append(nn.GELU())
                projs.append(nn.Linear(in_dim, out_dim))
                if len(projs) == 1:
                    return projs[0]
                return nn.Sequential(*projs)

            pred_steps = self.cfg.num_forward if not self.cfg.avg_num_forward else 1
            self.recon_proj = make_proj(cfg.embed_dim, cfg.embed_dim * pred_steps, cfg.recon_depth)
            self.recon_scale = 1 / math.sqrt(cfg.embed_dim) / pred_steps

            if cfg.pretrained_target_model_path is not None:
                assert self.cfg.ema_decay == 0
                state = checkpoint_utils.load_checkpoint_to_cpu(cfg.pretrained_target_model_path)
                teacher_cfg = state["cfg"]
                teacher_cfg.criterion = None
                teacher_cfg.lr_scheduler = None

                # teacher_cfg.task.data = cfg.data
                task = tasks.setup_task(teacher_cfg.task)
                model = task.build_model(teacher_cfg.model, from_checkpoint=True)

                model.remove_pretraining_modules()

                if "_ema" in state["model"]:
                    del state["model"]["_ema"]
                if "target_proj" in state["model"]:
                    del state["model"]["target_proj"]
                model.load_state_dict(state["model"], strict=True)
                for p in model.parameters():
                    p.requires_grad = False

                self.target_model = model

            elif self.cfg.ema_decay > 0:
                self.make_ema_teacher()

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.output_projection is not None:
            for k, _ in self.output_projection.named_parameters():
                skip_keys.add(f"output_projection.{k}")
        if self.adaptive_softmax is not None:
            for k, _ in self.adaptive_softmax.named_parameters():
                skip_keys.add(f"adaptive_softmax.{k}")
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.embed_positions.named_parameters():
                skip_keys.add(f"embed_positions.{k}")

        if self.cfg.ema_transformer_only:
            for k, _ in self.embed_tokens.named_parameters():
                skip_keys.add(f"embed_tokens.{k}")

        self.ema = EMAModule(
            self,
            ema_config,
            skip_keys=skip_keys,
        )
        print("made ema teacher")

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates == num_updates
        ):
            p = next(self.parameters())
            device = p.device
            dtype = p.dtype

            p = next(self.ema.model.parameters())
            ema_device = p.device
            ema_dtype = p.dtype

            if ema_device != device or ema_dtype != dtype:
                logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
                self.ema.model = self.ema.model.to(dtype=dtype, device=device)
                for k, p in self.ema.fp32_params.items():
                    self.ema.fp32_params[k] = p.to(device=device)

        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        if self.target_model is None:
            for k in list(state_dict.keys()):
                if k.startswith("target_model."):
                    del state_dict[k]

        k = prefix + "recon_proj.weight"
        if self.recon_proj is None and k in state_dict:
            del state_dict[k]
            del state_dict[prefix + "recon_proj.bias"]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def supported_targets(self):
        return {"future"}

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        return cls(cfg, task.target_dictionary)

    def build_output_projection(
        self, cfg: TransformerLMConfig, dictionary, embed_tokens
    ):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def build_layer(self, cfg):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, True)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    @staticmethod
    def compute_var(y):
        y = y.reshape(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def forward(
        self,
        src_tokens,
        target=None,
        src_lengths=None,
        features_only=False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        x, extra = self.extract_features(
            src_tokens,
            incremental_state=incremental_state,
        )

        lm_x = x.detach() if self.cfg.detach_lm else x

        if self.project_out_dim is not None:
            lm_x = self.project_out_dim(lm_x)

        lm_x = self.output_layer(lm_x)

        if features_only or target is None:
            return lm_x, extra

        lm_x = lm_x.view(-1, lm_x.size(-1))

        if self.adaptive_softmax is not None:
            logits, adap_target = self.adaptive_softmax(lm_x, target.view(-1))
            xentropy_loss = lm_x.new(1).zero_()

            for i in range(len(adap_target)):
                if adap_target[i] is not None:
                    assert adap_target[i].min() >= 0 and adap_target[i].max() <= logits[
                        i
                    ].size(1)
                    xentropy_loss += F.cross_entropy(
                        logits[i],
                        adap_target[i],
                        ignore_index=self.padding_idx,
                        reduction="sum",
                    )
        else:
            xentropy_loss = cross_entropy(
                lm_x,
                target.view(-1),
                ignore_index=self.padding_idx,
                reduction="none",
            )

        padding_mask = target.eq(self.padding_idx)

        result = {
            "losses": {
                "xentropy": xentropy_loss,
            },
            "sample_size": padding_mask.logical_not().sum(),
        }

        if self.cfg.average_top_k_layers > 0:
            x = self.recon_proj(x).float()

            with torch.no_grad():

                if self.ema is not None:
                    self.ema.model.eval()
                    _, teacher_res = self.ema.model.extract_features(target)
                elif self.target_model is not None:
                    self.target_model.eval()
                    if self.cfg.predict_teacher_targets:
                        _, teacher_res = self.target_model.extract_features(target, features_only=True, return_all_hiddens=True)
                    else:
                        _, teacher_res = self.target_model.extract_features(src_tokens, features_only=True, return_all_hiddens=True)
                else:
                    teacher_res = extra

                if self.cfg.use_fc_target:
                    y = teacher_res["fc_states"]
                else:
                    y = teacher_res["inner_states"]

                y = y[-self.cfg.average_top_k_layers :]

                if self.cfg.layer_norm_target_layer:
                    y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
                y = sum(y).float() / len(y)
                y = y.transpose(0, 1)

                if self.cfg.layer_norm_targets:
                    y = F.layer_norm(y, y.shape[-1:])

            if self.ema is None:
                x = x[:, :-1]
                y = y[:, 1:]
                if padding_mask is not None:
                    padding_mask = padding_mask[:, :-1] | padding_mask[:, 1:]

            if self.cfg.num_forward > 1:
                concat_pads = [padding_mask] if padding_mask is not None and padding_mask.any() else None
                if not self.cfg.avg_num_forward:
                    xs = x.chunk(self.cfg.num_forward, dim=-1)
                    concat_xs = [xs[0]]
                    for i in range(1, self.cfg.num_forward):
                        concat_xs.append(xs[i][:, :-i])
                        if concat_pads is not None:
                            concat_pads.append(padding_mask[:, :-i])
                    x = torch.cat(concat_xs, dim=1)

                concat_ys = [y]
                for i in range(1, self.cfg.num_forward):
                    concat_ys.append(y[:, i:])
                    if not self.cfg.avg_num_forward and concat_pads is not None:
                        concat_pads[i] = concat_pads[i] | padding_mask[:, i:]

                if not self.cfg.avg_num_forward:
                    y = torch.cat(concat_ys, dim=1)
                    if concat_pads is not None:
                        padding_mask = torch.cat(concat_pads, dim=1)
                else:
                    y = concat_ys[0].clone()
                    for i in range(1, len(concat_ys)):
                        y[:,:-i] += concat_ys[i]
                    y = y / self.cfg.num_forward


            recon_loss = F.mse_loss(x, y, reduction="none").sum(dim=-1)

            if padding_mask.any():
                recon_loss[padding_mask] = 0

            result["losses"]["regression"] = recon_loss.sum() * self.recon_scale

        return result

    def extract_features(
        self,
        src_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            src_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        fc_states: List[Optional[Tensor]] = []
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, fc_result = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            fc_states.append(fc_result)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": [attn], "inner_states": inner_states, "fc_states": fc_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_layer(self, args, no_encoder_attn=False):
        return super().build_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )
