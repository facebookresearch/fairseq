# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional
import copy
import logging
import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from omegaconf import II

from fairseq.data.data_utils import compute_mask_indices, compute_mask_indices_v2, compute_mask_indices_v3
from fairseq.dataclass import FairseqDataclass
from fairseq.modules import (
    EMAModule,
    EMAModuleConfig,
    Fp32LayerNorm,
    LayerNorm,
    SamePad,
    TransposeLast,
)
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
)
from fairseq.models.roberta.model import RobertaLMHead, RobertaClassificationHead
from fairseq.models.transformer import TransformerEncoder, TransformerConfig
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import index_put
from .utils import get_alibi, masked_alibi

logger = logging.getLogger(__name__)


@dataclass
class Data2VecTextConfig(FairseqDataclass):
    max_positions: int = II("task.tokens_per_sample")

    head_layers: int = 1

    transformer: TransformerConfig = TransformerConfig()

    load_checkpoint_heads: bool = field(
        default=False,
        metadata={"help": "(re-)register and load heads when loading checkpoints"},
    )

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    min_pred_var: float = field(
        default=0.01, metadata={"help": "stop training if prediction var falls below this"},
    )
    min_targ_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    # d2v2
    target_layerdrop: float = 0
    clone_batch: int = 1
    no_norm_if_avg_0: bool = False
    end_of_block_targets: bool = False
    affine_features_norm: bool = True
    layer_norm_pred: bool = False

    # MSN loss related
    msn_style: bool = False

    # wav2vec contrastive loss related
    w2v_style: bool = False

    # MAE related
    mae_style: bool = False
    mae_decoder_dim: int = 512
    mae_decoder_layers: int = 6
    mae_decoder_heads: int = 16
    mae_decoder_dropout: float = 0
    mae_ffn_factor: int = 4
    mae_decoder_learned_pos: bool = True
    mae_use_out_proj: bool = False
    mae_proj_is_decoder: bool = False  # NOTE: change from in_proj to proj
    mae_regr_is_decoder: bool = False  # NOTE change final_proj to regr
    mae_regr_separate: bool = False  # NOTE change final_proj to regr
    mae_encoder_decoder_attn: bool = False
    mae_pos_after_shrink: bool = False
    mae_mask_pct_future: float = 0
    mae_no_self_attn: bool = False
    mae_use_mask_emb: bool = False  # NOTE: use mask_emb instead of decoder embedding
    mae_use_rand_emb: bool = False  # NOTE: use random noise. This precedes mae_use_mask_emb
    mae_mask_noise_std: float = 0.

    mae_conv_decoder_layers: int = 0
    mae_conv_decoder_ln: bool = True
    mae_conv_decoder_act: bool = True
    mae_conv_decoder_kernel: int = 5
    mae_conv_decoder_groups: int = 1
    mae_conv_decoder_copy_input: bool = False

    mae_conv_decoder_residual: bool = False
    mae_conv_decoder_residual_avg: bool = False
    mae_conv_decoder_residual_act: bool = False
    mae_conv_decoder_residual_ln: bool = False

    mae_no_ffn: bool = False
    mae_no_norm1: bool = False
    mae_no_norm2: bool = False
    mae_no_prenorm: bool = False
    mae_no_enc_dec_resid: bool = False

    # mae_inverse_mask: bool = False

    # split_groups: int = 1
    # split_after_pos: bool = False
    # split_targets: bool = False
    # shrink_prob: float = 0

    # remask: bool = False  # NOTE: legacy. set task.skip_masking=True instead
    copy_mask: bool = False  # NOTE: for debugging, remasking with original ones
    random_token_prob: float = II("task.random_token_prob")
    leave_unmasked_prob: float = II("task.leave_unmasked_prob")
    mask_prob: float = II("task.mask_prob")
    mask_multiple_length: int = II("task.mask_multiple_length")
    mask_whole_words: bool = II("task.mask_whole_words")
    mask_stdev: float = II("task.mask_stdev")
    task_skip_masking: bool = II("task.skip_masking")
    remask_ver: int = 2
    idc_select_ver: int = 1
    num_mask_ver: int = 2

    seed: int = II("common.seed")

    require_same_masks: bool = False
    min_masks: int = 1

    use_alibi_encoder: bool = False
    use_alibi_decoder: bool = False
    log_mask_stat: bool = False

def get_annealed_rate(start, end, curr_step, total_steps):
    if curr_step >= total_steps:
        return end
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("data2vec_text", dataclass=Data2VecTextConfig)
class Data2VecTextModel(FairseqEncoderModel):
    def __init__(self, cfg: Data2VecTextConfig, encoder):
        super().__init__(encoder)
        self.cfg = cfg

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        encoder = Data2VecTextEncoder(cfg, task.source_dictionary, task.cfg.data)

        return cls(cfg, encoder)

    def forward(
        self,
        src_tokens,
        target_tokens=None,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        res = self.encoder(
            src_tokens, target_tokens, features_only, return_all_hiddens, **kwargs
        )

        if isinstance(res, tuple):
            x, extra = res
        else:
            return res

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=self.cfg.transformer.encoder.embed_dim,
            inner_dim=inner_dim or self.cfg.transformer.encoder.embed_dim,
            num_classes=num_classes,
            activation_fn="tanh",
            pooler_dropout=0,
        )

    @property
    def supported_targets(self):
        return {"self"}

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

            if self.encoder.regression_head is not None:
                if ".lm_head." in k:
                    new_k = k.replace(".lm_head.", ".regression_head.")
                    state_dict[new_k] = state_dict[k]
                    del state_dict[k]
            else:
                if ".regression_head." in k:
                    del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            or self.classification_heads is None
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if self.cfg.load_checkpoint_heads:
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if (
            hasattr(self, "classification_heads")
            and self.classification_heads is not None
            and len(self.classification_heads) > 0
        ):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

            for k in list(state_dict.keys()):
                if k.startswith(prefix + "encoder.lm_head.") or k.startswith(
                    prefix + "encoder.emb_head."
                ):
                    del state_dict[k]

            self.encoder.lm_head = None

        if hasattr(self.encoder, "target_model") and self.encoder.target_model is None:
            for k in list(state_dict.keys()):
                if k.startswith(prefix + "encoder.target_model."):
                    del state_dict[k]

        if (self.encoder.ema is None) and (prefix + "encoder._ema" in state_dict):
            del state_dict[prefix + "encoder._ema"]

    def remove_pretraining_modules(self, last_layer=None):
        self.encoder.lm_head = None
        self.encoder.regression_head = None
        self.encoder.ema = None
        self.classification_heads = None

        if last_layer is not None:
            self.encoder.sentence_encoder.layers = nn.ModuleList(
                l
                for i, l in enumerate(self.encoder.sentence_encoder.layers)
                if i <= last_layer
            )
            self.encoder.sentence_encoder.layer_norm = None


class Data2VecTextEncoder(FairseqEncoder):
    def __init__(self, cfg: Data2VecTextConfig, dictionary, task_data):
        super().__init__(dictionary)

        self.cfg = cfg
        if cfg.msn_style:
            raise NotImplementedError()
        elif cfg.w2v_style:
            raise NotImplementedError()
        if cfg.task_skip_masking:
            assert (
                cfg.mask_multiple_length >= 1
            ), f"invalid mask_multiple_length {cfg.mask_multiple_length}"
            assert (
                cfg.mask_prob > 0 and cfg.mask_prob < 1
            ), f"invalid mask_prob {cfg.mask_prob}"
            assert (
                cfg.random_token_prob == 0
            ), f"random_token_prob > 0 not supported yet"
            assert (
                cfg.leave_unmasked_prob == 0
            ), f"leave_unmasked_prob > 0 not supported yet"
            assert (
                cfg.mask_stdev == 0
            ), f"mask_stdev > 0 not supported yet"
            assert (
                cfg.mask_whole_words == False
            ), f"mask_whole_words not supported yet"
        assert (
            not cfg.mae_style or (cfg.task_skip_masking and cfg.require_same_masks)
        ), "setting task_skip_masking and require_same_masks is required for MAE"

        self.mask_idx = dictionary.index("<mask>")
        self.pad_idx = dictionary.pad()
        assert self.mask_idx != dictionary.unk(), dictionary.symbols
        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_scale = cfg.loss_scale

        embed_tokens = self.build_embedding(
            len(dictionary), cfg.transformer.encoder.embed_dim, dictionary.pad()
        )
        self.sentence_encoder = self.build_encoder(cfg, dictionary, embed_tokens)
        self.ema = None
        self.regression_head = self.build_regression_head(cfg)
        self.alibi_biases = {}

        self.clone_hash = [
            int(hash((self.cfg.seed, ind)) % 1e10)
            for ind in range(self.cfg.clone_batch - 1)
        ]
        self.clone_hash = torch.tensor([0] + self.clone_hash).long().view(1, -1)

        self.decoder = None
        if cfg.mae_style:
            if cfg.mae_conv_decoder_layers > 0 or cfg.mae_use_mask_emb:
                self.decoder_mask_emb = (
                    nn.Parameter(torch.FloatTensor(cfg.mae_decoder_dim).uniform_())
                )

            if cfg.mae_conv_decoder_layers > 0:
                self.decoder = self.build_conv_decoder(cfg)
            else:
                # TODO: use dummy embed with only <mask> and reserved
                decoder_embed_tokens = self.build_embedding(
                    len(dictionary), cfg.mae_decoder_dim, dictionary.pad()
                )
                self.decoder = self.build_decoder(
                    cfg, dictionary, decoder_embed_tokens
                )
            (
                self.decoder_in_proj, self.decoder_out_proj
            ) = self.build_decoder_projections(cfg)
            self.group_parameters(cfg)

        self.num_updates = 0
        self.epoch = 0

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, cfg, dictionary, embed_tokens):
        encoder = TransformerEncoder(cfg.transformer, dictionary, embed_tokens, return_fc=True)
        encoder.apply(init_bert_params)
        return encoder

    def build_regression_head(self, cfg):
        assert cfg.head_layers >= 1

        encoder_dim = cfg.transformer.encoder.embed_dim
        decoder_dim = cfg.mae_decoder_dim
        if not cfg.mae_style or cfg.mae_use_out_proj:
            curr_dim = encoder_dim
        else:
            curr_dim = decoder_dim
        projs = []
        for i in range(cfg.head_layers - 1):
            next_dim = encoder_dim * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim

        projs.append(nn.Linear(curr_dim, encoder_dim))
        if cfg.layer_norm_pred:
            projs.append(Fp32LayerNorm(encoder_dim))
        regression_head = nn.Sequential(*projs)
        return regression_head

    def build_decoder(self, cfg, dictionary, embed_tokens):
        t_cfg = copy.deepcopy(cfg)
        t_cfg.transformer.encoder.layers = cfg.mae_decoder_layers
        t_cfg.transformer.encoder.embed_dim = cfg.mae_decoder_dim
        t_cfg.transformer.encoder.ffn_embed_dim = cfg.mae_decoder_dim * cfg.mae_ffn_factor
        t_cfg.transformer.encoder.attention_heads = cfg.mae_decoder_heads
        t_cfg.transformer.encoder.layerdrop = 0.
        t_cfg.transformer.encoder.learned_pos = cfg.mae_decoder_learned_pos
        cross_dim = (
            cfg.transformer.encoder.embed_dim
            if cfg.mae_encoder_decoder_attn
            else None
        )
        decoder = TransformerEncoder(
            t_cfg.transformer,
            dictionary,
            embed_tokens,
            return_fc=True,
            cross_dim=cross_dim,
            self_attn=(not cfg.mae_no_self_attn),
            ffn=(not cfg.mae_no_ffn),
            norm1=(not cfg.mae_no_norm1),
            norm2=(not cfg.mae_no_norm2),
            cross_resid=(not cfg.mae_no_enc_dec_resid),
        )
        decoder.apply(init_bert_params)
        return decoder

    def build_conv_decoder(self, cfg):
        def make_block():
            conv = nn.Conv1d(
                cfg.mae_decoder_dim,
                cfg.mae_decoder_dim,
                kernel_size=cfg.mae_conv_decoder_kernel,
                padding=cfg.mae_conv_decoder_kernel // 2,
                groups=cfg.mae_conv_decoder_groups,
            )
            pad = SamePad(cfg.mae_conv_decoder_kernel)
            norm = []
            if cfg.mae_conv_decoder_ln:
                norm = [
                    TransposeLast(),
                    LayerNorm(cfg.mae_decoder_dim, elementwise_affine=False),
                    TransposeLast(),
                ]
            act = nn.GELU() if cfg.mae_conv_decoder_act else nn.Identity()
            layers = [conv, pad] + norm + [act]
            return nn.Sequential(*layers)

        decoder = nn.Sequential(
            *[make_block() for _ in range(cfg.mae_conv_decoder_layers)]
        )
        return decoder

    def build_decoder_projections(self, cfg):
        encoder_dim = cfg.transformer.encoder.embed_dim
        decoder_dim = cfg.mae_decoder_dim
        decoder_in_proj = None
        decoder_out_proj = None
        if (
            not cfg.mae_encoder_decoder_attn
            and encoder_dim != decoder_dim
            and cfg.mae_conv_decoder_layers == 0
        ):
            decoder_in_proj = nn.Linear(encoder_dim, decoder_dim)
        if (
            cfg.mae_use_out_proj
            or (
                cfg.mae_conv_decoder_layers > 0
                and encoder_dim != decoder_dim
            )
        ):
            decoder_out_proj = nn.Linear(decoder_dim, encoder_dim)
        return decoder_in_proj, decoder_out_proj

    def group_parameters(self, cfg):
        for p in self.parameters():
            p.param_group = "encoder"
        for p in self.decoder.parameters():
            p.param_group = "decoder"

        if cfg.mae_proj_is_decoder:
            if self.decoder_in_proj is not None:
                for p in self.decoder_in_proj.parameters():
                    p.param_group = "decoder"
            if self.decoder_out_proj is not None:
                for p in self.decoder_out_proj.parameters():
                    p.param_group = "decoder"

        if cfg.mae_regr_is_decoder:
            for p in self.regression_head.parameters():
                p.param_group = "decoder"
        elif cfg.mae_regr_separate:
            for p in self.regression_head.parameters():
                p.param_group = "final"

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    @torch.no_grad()
    def get_alibi_bias(self, heads, batch_size, time_steps, dtype, device):
        buffered = self.alibi_biases.get(heads, None)

        target_size = heads * batch_size
        if (
            buffered is None
            or buffered.size(0) < target_size
            or buffered.size(1) < time_steps
            or buffered.dtype != dtype
            or buffered.device != device
        ):
            bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
            bn = (
                max(target_size, buffered.size(0) if buffered is not None else 0)
                // heads
            )

            buffered = (
                get_alibi(bt, heads)
                .to(dtype=dtype, device=device)
                .repeat(bn, 1, 1)
            )

            self.alibi_biases[heads] = buffered

        return buffered[:target_size, :time_steps, :time_steps]

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_transformer_layers_only:
            for k, _ in self.sentence_encoder.embed_tokens.named_parameters():
                skip_keys.add(f"embed_tokens.{k}")
            for k, _ in self.sentence_encoder.embed_positions.named_parameters():
                skip_keys.add(f"embed_positions.{k}")
            if self.sentence_encoder.layernorm_embedding is not None:
                for (
                    k,
                    _,
                ) in self.sentence_encoder.layernorm_embedding.named_parameters():
                    skip_keys.add(f"layernorm_embedding.{k}")
            if self.sentence_encoder.layer_norm is not None:
                for k, _ in self.sentence_encoder.layer_norm.named_parameters():
                    skip_keys.add(f"layernorm_embedding.{k}")

        self.ema = EMAModule(
            self.sentence_encoder,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.regression_head is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
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
                self.ema.step(self.sentence_encoder)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params
        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        src_tokens,
        target_tokens=None,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        src_id=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        assert features_only or target_tokens is not None

        padding_mask = src_tokens.eq(self.pad_idx)
        unmasked_tokens = src_tokens.clone().detach()
        if target_tokens is None:
            mask_mask = None
        else:
            mask_mask = target_tokens.ne(self.pad_idx)
            unmasked_tokens[mask_mask] = target_tokens[mask_mask]
            unmasked_tokens = unmasked_tokens.detach()

        if not unmasked_tokens.ne(self.mask_idx).all():
            print((
                "mask positions do not match between src_tokens and target_tokens. "
                "something went very wrong...\n"
                f"src_id = {src_id}\n"
                f"matched = {unmasked_tokens.ne(self.mask_idx).all(-1)}"
            ))
            raise

        if mask_mask is not None:
            assert (
                torch.logical_or(~padding_mask, ~mask_mask).all()
            ), "mask_mask has True at padded some token(s)"


        assert (
            self.cfg.task_skip_masking or self.cfg.clone_batch == 1
        ), f"task_skip_masking is required when clone_batch > 1 ({self.cfg.clone_batch})"
        assert (
            not self.cfg.copy_mask or self.cfg.clone_batch == 1
        ), f"copy_mask is incompatible with clone_batch > 1"

        if self.cfg.task_skip_masking and mask_mask is not None:
            mask_mask_copy = mask_mask.detach().cpu().numpy()
            src_tokens, src_padding_mask, mask_mask = self.clone_and_remask(
                unmasked_tokens, padding_mask, mask_mask_copy, src_id
            )

        if (
            self.cfg.log_mask_stat and
            mask_mask is not None and
            self.cfg.clone_batch == 1
        ):
            mask_stat = self.log_mask_stat(mask_mask, padding_mask)

        # TODO: annotate dimensions
        x, extra = self.extract_encoder(
            src_tokens,
            mask_mask,
            features_only,
            return_all_hiddens,
        )

        if features_only:
            return x, extra

        if self.decoder is not None:
            x = self.extract_decoder(x, src_tokens, mask_mask)
        else:
            x = x[mask_mask]

        x = self.regression_head(x).float()
        y = self.extract_ema(unmasked_tokens, mask_mask).float()

        x_var = self.compute_var(x).item()
        y_var = self.compute_var(y).item()
        self.check_var(x_var, y_var)

        result = {}
        loss, sample_size = self.compute_d2v_loss(x, y)
        result["losses"] = {"main": loss}
        result["sample_size"] = sample_size
        result["logs"] = {
            "ema_decay": self.ema.get_decay() * 1000,
            "pred_var": x_var,
            "targ_var": y_var,
        }
        if self.cfg.log_mask_stat and mask_mask is not None:
            result["logs"]["avg_masked"] = mask_stat[0]
            result["logs"]["min_masked"] = mask_stat[1]
            result["logs"]["max_masked"] = mask_stat[2]
            result["logs"]["masked_ratio"] = mask_stat[3]
            result["logs"]["masked_min_span"] = mask_stat[4]
            result["logs"]["masked_max_span"] = mask_stat[5]
            result["logs"]["mask_idc_sum"] = mask_mask.nonzero().sum().item()
            result["logs"]["src_id"] = src_id.sum().item()
            result["logs"]["min_leng"] = (~padding_mask).sum(1).min().item()
        # print(src_id.detach().cpu().numpy())

        return result

    def clone_and_remask(self, unmasked_tokens, padding_mask, orig_mask_mask, src_id):
        if self.cfg.clone_batch > 1:
            unmasked_tokens = unmasked_tokens.repeat_interleave(self.cfg.clone_batch, 0)
            padding_mask = padding_mask.repeat_interleave(self.cfg.clone_batch, 0)
            src_id = src_id.repeat_interleave(self.cfg.clone_batch, 0)
            src_id = src_id.view(-1, self.cfg.clone_batch) + self.clone_hash.to(src_id)
            src_id = src_id.view(-1)

        if self.cfg.remask_ver == 1:
            mask_mask = compute_mask_indices(
                unmasked_tokens.size(),
                padding_mask,
                self.cfg.mask_prob,
                self.cfg.mask_multiple_length,
                min_masks=self.cfg.min_masks,
                require_same_masks=self.cfg.require_same_masks,
                seed=self.cfg.seed,
                epoch=self.epoch,
                indices=src_id,
                idc_select_ver=self.cfg.idc_select_ver,
                num_mask_ver=self.cfg.num_mask_ver,
            )

        elif self.cfg.remask_ver == 2:
            mask_mask = compute_mask_indices_v2(
                unmasked_tokens.size(),
                padding_mask,
                self.cfg.mask_prob,
                self.cfg.mask_multiple_length,
                min_masks=self.cfg.min_masks,
                require_same_masks=self.cfg.require_same_masks,
                seed=self.cfg.seed,
                epoch=self.epoch,
                indices=src_id,
            )
        elif self.cfg.remask_ver == 3:
            mask_mask = compute_mask_indices_v3(
                unmasked_tokens.size(),
                padding_mask,
                self.cfg.mask_prob,
                self.cfg.mask_multiple_length,
                min_masks=self.cfg.min_masks,
                require_same_masks=self.cfg.require_same_masks,
                seed=self.cfg.seed,
                epoch=self.epoch,
                indices=src_id,
            )
        else:
            raise ValueError(f"remask_ver {self.cfg.remask_ver} not supported")

        if self.cfg.copy_mask:
            mask_mask[:] = orig_mask_mask
        mask_mask = torch.from_numpy(mask_mask).to(unmasked_tokens.device)
        src_tokens = unmasked_tokens.clone().detach()
        src_tokens[mask_mask] = self.mask_idx
        src_tokens = src_tokens.detach()
        assert (
            torch.logical_or(~padding_mask, ~mask_mask).all()
        ), "mask_mask has True at padded some token(s)"
        return src_tokens, padding_mask, mask_mask

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        fc_states = encoder_out["fc_results"] if return_all_hiddens else None

        return features, {
            "inner_states": inner_states,
            "fc_states": fc_states,
            "encoder_embedding": encoder_out["encoder_embedding"][0],
        }

    def extract_encoder(self, src_tokens, mask_mask, features_only, return_all_hiddens):
        encoder_x, _ = self.sentence_encoder.forward_embedding(src_tokens)
        encoder_src_tokens = src_tokens

        doing_mae = (
            self.cfg.mae_style
            and not features_only
            and mask_mask is not None
        )
        if doing_mae:
            assert (mask_mask.sum(1) == mask_mask[0].sum()).all()
            B = mask_mask.size(0)
            T = mask_mask.size(1) - mask_mask[0].sum()
            encoder_x = encoder_x[~mask_mask].view(B, T, -1)
            # encoder_padding_mask = padding_mask[~mask_mask].view(B, T)
            encoder_src_tokens = src_tokens[~mask_mask].view(B, T)

        alibi_bias = None
        if self.cfg.use_alibi_encoder:
            orig_B, orig_T = src_tokens.size()
            alibi_bias = self.get_alibi_bias(
                self.cfg.transformer.encoder.attention_heads,
                batch_size=orig_B,
                time_steps=orig_T,
                dtype=encoder_x.dtype,
                device=encoder_x.device,
            )
            if doing_mae:
                alibi_bias = masked_alibi(alibi_bias, (~mask_mask), orig_B, orig_T)

        # encoder
        encoder_out = self.sentence_encoder.forward_layers(
            encoder_src_tokens,
            encoder_x,
            return_all_hiddens=return_all_hiddens,
            alibi_bias=alibi_bias,
        )
        # T x B x C -> B x T x C
        x = encoder_out["encoder_out"][0].transpose(0, 1)
        extra = {
            "inner_states": encoder_out["encoder_states"],
            "fc_states": encoder_out["fc_results"],
            "encoder_embedding": encoder_out["encoder_embedding"][0],
        }
        return x, extra

    def extract_decoder(self, encoder_out, src_tokens, mask_mask):
        x = encoder_out
        if self.cfg.mae_decoder_dropout > 0:
            x = F.dropout(x, self.cfg.mae_decoder_dropout, inplace=True)

        decoder_input = x.new_zeros(
            (src_tokens.size(0),  src_tokens.size(1), self.cfg.mae_decoder_dim)
        )

        # fill unmasked frames
        decoder_input_u = x
        if self.decoder_in_proj is not None:
            decoder_input_u = self.decoder_in_proj(x)
        decoder_input_u = decoder_input_u.reshape(-1, decoder_input_u.size(-1))
        decoder_input = index_put(
            decoder_input,
            ~mask_mask,
            decoder_input_u,
        )

        # fill masked frames
        if self.cfg.mae_use_rand_emb:
            decoder_input_m = decoder_input.new_zeros(
                (mask_mask.sum(), decoder_input.size(-1))
            ).normal_(0, self.cfg.mae_mask_noise_std)
        elif self.cfg.mae_conv_decoder_layers > 0 or self.cfg.mae_use_mask_emb:
            decoder_input_m = self.decoder_mask_emb
        else:
            decoder_input_m = self.decoder.embed_tokens(
                src_tokens.new_zeros((1,)).fill_(self.mask_idx)
            ).view(1, -1)
        decoder_input = index_put(
            decoder_input,
            mask_mask,
            decoder_input_m,
        )

        # get alibi bias
        alibi_bias = None
        if self.cfg.use_alibi_decoder:
            orig_B, orig_T = src_tokens.size()
            alibi_bias = self.get_alibi_bias(
                self.cfg.mae_decoder_heads,
                batch_size=orig_B,
                time_steps=orig_T,
                dtype=decoder_input.dtype,
                device=decoder_input.device,
            )

        # add position embedding (transformer only)
        if self.cfg.mae_conv_decoder_layers == 0:
            decoder_input, _ = self.decoder.forward_embedding(
                src_tokens, token_embedding=decoder_input
            )

        if self.cfg.mae_conv_decoder_layers > 0:
            decoder_out = decoder_input.transpose(1, 2)
            if self.cfg.mae_conv_decoder_copy_input:
                for layer_i, layer in enumerate(self.decoder):
                    if i > 0:
                        decoder_out = decoder_out.transpose(1, 2)
                        decoder_out = index_put(
                            decoder_out, ~mask_mask, decoder_input_u
                        )
                        decoder_out = decoder_out.transpose(1, 2)
                    decoder_out = layer(decoder_out)
            elif self.cfg.mae_conv_decoder_residual:
                decoder_res = decoder_out
                for layer_i, layer in enumerate(self.decoder):
                    decoder_out = layer(decoder_out)
                    decoder_out = self.add_residual(decoder_out, decoder_res)
                    decoder_res = decoder_out
            else:
                decoder_out = self.decoder(decoder_out)
            decoder_out = decoder_out.transpose(1, 2)
        elif not self.cfg.mae_encoder_decoder_attn:
            assert (
                not self.cfg.mae_no_self_attn
            ), "need self-attention if no cross-attention"
            decoder_out = self.decoder.forward_layers(
                src_tokens,
                decoder_input,
                alibi_bias=alibi_bias,
            )
        else:
            assert (mask_mask.sum(1) == mask_mask[0].sum()).all()
            B = mask_mask.size(0)
            T = mask_mask[0].sum()
            decoder_input = decoder_input[mask_mask].view(B, T, -1)
            decoder_src_tokens = src_tokens[mask_mask].view(B, T)
            if alibi_bias is not None:
                alibi_bias = masked_alibi(alibi_bias, mask_mask, orig_B, orig_T)
            decoder_out = self.decoder.forward_layers(
                decoder_src_tokens,
                decoder_input,
                cross_out=x.transpose(0, 1),
                cross_padding_mask=None,  # TODO
                alibi_bias=alibi_bias,
            )

        if self.cfg.mae_conv_decoder_layers == 0:
            x = decoder_out["encoder_out"][0].transpose(0, 1)
        else:
            x = decoder_out

        if (
            not self.cfg.mae_encoder_decoder_attn
            or self.cfg.mae_conv_decoder_layers > 0
        ):
            x = x[mask_mask]
        else:
            x = x.reshape(-1, x.size(-1))

        if self.decoder_out_proj is not None:
            x = self.decoder_out_proj(x)

        return x

    def add_residual(self, out, res):
        if res is None:
            return out

        out = out + res
        if self.cfg.mae_conv_decoder_residual_avg:
            out = out / 2
        if self.cfg.mae_conv_decoder_residual_ln:
            out = out.transpose(1, 2)
            out = F.layer_norm(out.float(), out.shape[-1:]).type_as(out)
            out = out.transpose(1, 2)
        if self.cfg.mae_conv_decoder_residual_act:
            out = F.gelu(out)
        return out

    @torch.no_grad()
    def extract_ema(self, unmasked_tokens, mask_mask):
        # use EMA parameter as the teacher
        self.ema.model.eval()

        alibi_bias = None
        if self.cfg.use_alibi_encoder:
            orig_B, orig_T = unmasked_tokens.size()
            alibi_bias = self.get_alibi_bias(
                self.cfg.transformer.encoder.attention_heads,
                batch_size=orig_B,
                time_steps=orig_T,
                dtype=self.ema.model.embed_tokens.weight.dtype,
                device=unmasked_tokens.device,
            )

        encoder_out = self.ema.model(
            unmasked_tokens,
            return_all_hiddens=True,
            alibi_bias=alibi_bias,
        )
        y = encoder_out["fc_results"]

        y = y[-self.average_top_k_layers :]

        permuted = False
        if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
            y = [tl.permute(1, 2, 0) for tl in y]  # TBC -> BCT
            permuted = True

        if self.cfg.batch_norm_target_layer:
            y = [
                F.batch_norm(
                    tl.float(), running_mean=None, running_var=None, training=True
                )
                for tl in y
            ]

        if self.cfg.instance_norm_target_layer:
            y = [F.instance_norm(tl.float()) for tl in y]

        if permuted:
            y = [tl.transpose(1, 2) for tl in y]  # BCT -> BTC

        if self.cfg.layer_norm_target_layer:
            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]

        y = sum(y) / len(y)

        if not permuted:
            y = y.transpose(0, 1)

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y.float(), y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        if self.cfg.clone_batch > 1:
            y = y.repeat_interleave(self.cfg.clone_batch, 0)
        y = y[mask_mask]

        return y

    def compute_d2v_loss(self, x, y):
        """x y are N-by-D tensors, where N is the number of masked frames in the batch"""
        sz = x.size(-1)
        if self.cfg.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.cfg.loss_beta
            ).sum(dim=-1)

        if self.loss_scale > 0:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)
        loss = loss * scale

        sample_size = loss.size(0)
        # TODO: data2vec_audio does not sum when loss_dropout = 0
        loss = loss.sum()
        return loss, sample_size


        # logging other values
        other_logs = {
            "ema_decay": self.ema.get_decay() * 1000
        }
        result["logs"] = other_logs

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
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

    def check_var(self, pred_var, targ_var):
        if self.num_updates > 5000 and targ_var < self.cfg.min_targ_var:
            msg = f"target var is {targ_var} < {self.cfg.min_targ_var}, exiting"
            logger.error(msg)
            raise Exception(msg)

        if self.num_updates > 5000 and pred_var < self.cfg.min_pred_var:
            msg = f"prediction var is {pred_var} < {self.cfg.min_pred_var}, exiting"
            logger.error(msg)
            raise Exception(msg)

    @staticmethod
    def log_mask_stat(mask_mask, padding_mask=None):
        num_masked = mask_mask.sum(1).float()
        avg_masked = num_masked.mean().item()
        min_masked = num_masked.min().item()
        max_masked = num_masked.max().item()
        avg_leng = (
            mask_mask.size(1) if padding_mask is None
            else (~padding_mask).sum(1).float().mean()
        )
        masked_ratio = avg_masked / avg_leng.item()
        max_span = 0
        min_span = mask_mask.size(1) + 1
        for arr in mask_mask:
            cur_span = 0
            for val in arr:
                if val:
                    cur_span += 1
                elif cur_span != 0:
                    max_span = max(max_span, cur_span)
                    min_span = min(min_span, cur_span)
                    cur_span = 0
            if cur_span != 0:
                max_span = max(max_span, cur_span)
                min_span = min(min_span, cur_span)
        return avg_masked, min_masked, max_masked, masked_ratio, min_span, max_span

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.cfg.max_positions
