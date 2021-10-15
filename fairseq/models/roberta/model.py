# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import RobertaHubInterface


logger = logging.getLogger(__name__)


@dataclass
class RobertaModelConfig(FairseqDataclass):
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    pooler_activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="tanh", metadata={"help": "activation function to use for the pooler layer"}
    )
    encoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each encoder block"}
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    pooler_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the masked_lm pooler layers"}
    )
    max_positions: int = field(
        default=512, metadata={"help": "number of positional embeddings to learn"}
    )
    max_source_positions: int = field(
        default=512, metadata={"help": "number of source positional embeddings to learn"}
    )
    no_token_positional_embeddings: bool = field(
        default=False, metadata={"help": "don't use positional embeddings for tokens"}
    )
    encoder_learned_pos: bool = field(
        default=True, metadata={"help": "learn positional encodings in the encoder"}
    )
    encoder_normalize_before: bool = field(
        default=False, metadata={"help": "normalize before encoder"}
    )
    no_scale_embedding: bool = field(
        default=True, metadata={"help": "don't scale embeddings?"}
    )
    load_checkpoint_heads: bool = field(
        default=False, metadata={"help": "(re-)register and load heads when loading checkpoints"}
    )
    untie_weights_roberta: bool = field(
        default=False, metadata={"help": "Untie weights between embeddings and classifiers in RoBERTa"}
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "use adaptive inputs"}
    )

    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for encoder"}
    )
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
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
    # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
    spectral_norm_classification_head: bool = field(
        default=False, metadata={"help": "Apply spectral normalization on the classification head"}
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
        }
    )
    # args for MoE training
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    # Mixture of Expert Layer arguments
    moe_freq: int = field(
        default=0,
        metadata={
            "help": "Frequency at which we insert MoE Transformer layers"
        },
    )
    moe_expert_count: int = field(
        default=0,
        metadata={
            "help": "Number of experts in each MoE Layer"
        }
    )
    moe_gating_use_fp32: bool = field(
        default=False,
        metadata={
            "help": "Use FP32 computations in MoE top2 gating function"
        }
    )
    moe_second_expert_policy: str = field(
        default='sampling',
        metadata={
            "help": "policy for second expert, options: all/sampling/random"
        }
    )
    moe_normalize_gate_prob_before_dropping: bool = field(
        default=False,
        metadata={
            "help": 'whether to normalize gate probs before or after dropping experts for capacity and randomization'
        }
    )
    moe_expert_ffn_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE expert FFN dimension"
        }
    )
    moe_top1_expert: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use top1 gate instead of top2"
        }
    )
    moe_eval_capacity_token_fraction: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "Default: 0.25, Fraction of tokens as capacity during validation, if set to negative, use same as training. range: (0.0, 1.0]."
        }
    )
    moe_normalize_expert_grad: Optional[str] = field(
        default='sqrt_num_experts',
        metadata={
            "help": "Divide expert gradients by (1) 'num_experts' (2) 'sqrt_num_experts'"
        }
    )
    use_moe_pad_mask: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Don't route padding tokens to any expert",
        }
    )
    record_a2a_perf_stats: Optional[bool] = field(
        default=False, metadata={"help": "records all to all perf stats during distributed training"}
    )
    dummy_a2a: Optional[bool] = field(
        default=False, metadata={"help": "By passes all to all during distributed training by returning the input buffer as output"}
    )

    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    tpu: bool = II("common.tpu")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    fp16: bool = II("common.fp16")
    fp16_no_flatten_grads: bool = II("common.fp16_no_flatten_grads")
    ddp_backend: str = II("distributed_training.ddp_backend")
    world_size: int = II("distributed_training.distributed_world_size")
    distributed_rank: int = II("distributed_training.distributed_rank")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    task: str = II("task")
    use_stable_embedding: Optional[bool] = field(
        default=False,
        metadata={"help": 'Use bitsandbytes StableEmbeddingLayer which saves embedding state in fp32',
                  'argparse_alias': "--stable-emb"}
    )
    scale_fc: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Layernorm between fc"
        }
    )
    scale_attn: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Layernorm after attn"
        }
    )
    scale_heads: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Early-bert attention head pruning coeff"
        }
    )
    scale_resids: Optional[bool] = field(default=False, metadata={"help": 'Learn a scale coefficient for each attention head'})
    scale_first_resids: Optional[bool] = field(default=False,
                                               metadata={"help": 'Learn a scale coefficient for each attention head'})
    use_fused_softmax: bool = field(
        default=False, metadata={"help": "use Megatron softmax kernel"}
    )



@register_model("roberta", dataclass=RobertaModelConfig)
class RobertaModel(FairseqEncoderModel):
    @classmethod
    def hub_models(cls):
        return {
            "roberta.base": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
            "roberta.large": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz",
            "roberta.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz",
            "roberta.large.wsc": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz",
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)

        self.args=args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

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
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    @property
    def supported_targets(self):
        return {"self"}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )

        logger.info(x["args"])
        return RobertaHubInterface(x["args"], x["task"], x["models"][0])

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

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if self.args.task == 'sentence_prediction' and 'lm_head' in k:
                keys_to_delete.append(k)

            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
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
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        # set any missing default values
        base_architecture(args)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)
        if self.args.task != 'sentence_prediction':
            self.lm_head = self.build_lm_head(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=(
                    self.sentence_encoder.embed_tokens.weight
                    if not args.untie_weights_roberta
                    else None
                ),
            )

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused
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
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        l_aux = encoder_out["l_aux"]
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {"inner_states": inner_states, "l_aux": l_aux}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.use_fused_softmax = getattr(args, "use_fused_softmax", False)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )


@register_model_architecture("roberta", "roberta_prenorm")
def roberta_prenorm_architecture(args):
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    base_architecture(args)


@register_model_architecture("roberta", "roberta_base")
def roberta_base_architecture(args):
    base_architecture(args)


@register_model_architecture("roberta", "roberta_large")
def roberta_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("roberta", "xlm")
def xlm_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1280 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)
