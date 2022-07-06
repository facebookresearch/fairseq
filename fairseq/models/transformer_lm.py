# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    Embedding,
    TransformerDecoder,
)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from fairseq.utils import safe_getattr, safe_hasattr

logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class TransformerLanguageModelConfig(FairseqDataclass):
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

    moe_eom: float = field(default=0.0, metadata={"help": "MoE expert output masking"})
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_decoder_final_norm: bool = field(
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
    share_decoder_input_output_embed: bool = field(
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
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_learned_sinusoidal: bool = field(
        default=False,
        metadata={
            "help": "use learned positional embeddings init with sinusoidal in the decoder"
        },
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
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
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
    # Mixture of Expert Layer arguments
    alternate_decoder_ffn_embed_dim: int = field(
        default=0,
        metadata={"help": "decoder FFN embed dim of alternate decoder layers"},
    )
    moe_freq: int = field(
        default=0,
        metadata={"help": "Frequency at which we insert MoE Transformer layers"},
    )
    moe_expert_count: int = field(
        default=0, metadata={"help": "Number of experts in each MoE Layer"}
    )
    moe_gating_use_fp32: bool = field(
        default=False,
        metadata={"help": "Use FP32 computations in MoE top2 gating function"},
    )
    moe_second_expert_policy: str = field(
        default="sampling",
        metadata={"help": "policy for second expert, options: all/sampling/random"},
    )
    moe_normalize_gate_prob_before_dropping: bool = field(
        default=False,
        metadata={
            "help": "whether to normalize gate probs before or after dropping experts for capacity and randomization"
        },
    )
    moe_local_drop: float = field(
        default=0.0,
        metadata={"help": "Probability of gating drop out to local experts"},
    )
    moe_expert_ffn_dim: Optional[int] = field(
        default=None, metadata={"help": "MoE expert FFN dimension"}
    )
    moe_top1_expert: Optional[bool] = field(
        default=False, metadata={"help": "Use top1 gate instead of top2"}
    )
    moe_eval_capacity_token_fraction: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "Default: 0.25, Fraction of tokens as capacity during validation, if set to negative, use same as training. range: (0.0, 1.0]."
        },
    )
    moe_eval_capacity_max_seqlen: bool = field(
        default=False,
        metadata={
            "help": "Multiply the MoE capacity factor by the max sequence length instead of the current sequence length to get the effective expert capacity during validation"
        },
    )
    moe_normalize_expert_grad: Optional[str] = field(
        default="world_size",
        metadata={
            "help": "Divide expert gradients by (1) 'world_size' (2) 'sqrt_world_size'"
        },
    )
    use_moe_pad_mask: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Don't route padding tokens to any expert",
        },
    )
    record_a2a_perf_stats: Optional[bool] = field(
        default=False,
        metadata={"help": "records all to all perf stats during distributed training"},
    )
    dummy_a2a: Optional[bool] = field(
        default=False,
        metadata={
            "help": "By passes all to all during distributed training by returning the input buffer as output"
        },
    )
    moe_batch_prioritized_routing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if true orders token by the gate prob before capacity dropping."
        },
    )
    moe_cmr: Optional[bool] = field(
        default=False, metadata={"help": "If true enables CMR gating"}
    )
    cmr_log_lang_gates: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to log per-lang fraction of tokens routed to MOE vs. CMR"
        },
    )
    cmr_gate_drop: Optional[float] = field(
        default=0.0, metadata={"help": "CMR gate dropout rate"}
    )
    # NormFormer
    scale_fc: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Insert LayerNorm between fully connected layers",
        },
    )
    scale_attn: Optional[bool] = field(
        default=False, metadata={"help": "Insert LayerNorm after attention"}
    )
    scale_heads: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each attention head"},
    )
    scale_heads_inside: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Learn a scale coefficient for each attention head. Doing the math inside the attention head is slower"
        },
    )
    scale_resids: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each residual connection"},
    )
    use_stable_embedding: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use bitsandbytes StableEmbeddingLayer which saves embedding state in fp32",
            "argparse_alias": "--stable-emb",
        },
    )
    # options from other parts of the config

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
    # ALiBi
    alibi: bool = field(
        default=False,
        metadata={
            "help": "use the ALiBi position method instead of regular position embeddings"
        },
    )
    use_fused_softmax: bool = field(
        default=False, metadata={"help": "use Megatron softmax kernel"}
    )

    no_emb_dropout: Optional[bool] = field(
        default=False, metadata={"help": "Avoid emb dropout for decoder"}
    )

    init_model_on_gpu: Optional[bool] = field(
        default=False, metadata={"help": "Initialize model on GPUs."}
    )
    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    fp16: bool = II("common.fp16")
    fp16_no_flatten_grads: bool = II("common.fp16_no_flatten_grads")
    ddp_backend: str = II("distributed_training.ddp_backend")
    world_size: int = II("distributed_training.distributed_world_size")
    distributed_rank: int = II("distributed_training.distributed_rank")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    use_tutel_moe: bool = II("common.use_tutel_moe")


@register_model("transformer_lm", dataclass=TransformerLanguageModelConfig)
class TransformerLanguageModel(FairseqLanguageModel):
    @classmethod
    def hub_models(cls):
        def moses_fastbpe(path):
            return {"path": path, "tokenizer": "moses", "bpe": "fastbpe"}

        def spm(path):
            return {"path": path, "tokenizer": "space", "bpe": "sentencepiece"}

        return {
            "transformer_lm.gbw.adaptive_huge": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2",
            "transformer_lm.wiki103.adaptive": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2",
            "transformer_lm.wmt19.en": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2"
            ),
            "transformer_lm.wmt19.de": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2"
            ),
            "transformer_lm.wmt19.ru": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2"
            ),
            "transformer_lm.wmt20.en": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.en.tar.gz"
            ),
            "transformer_lm.wmt20.ta": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.ta.tar.gz"
            ),
            "transformer_lm.wmt20.iu.news": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.iu.news.tar.gz"
            ),
            "transformer_lm.wmt20.iu.nh": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.iu.nh.tar.gz"
            ),
        }

    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if safe_getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = safe_getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        if getattr(args, "moe_freq", 0) > 0 and (
            getattr(args, "fp16", False)
            and not getattr(args, "memory_efficient_fp16", False)
            and getattr(args, "ddp_backend", None) != "fully_sharded"
        ):
            assert (
                args.fp16_no_flatten_grads
            ), "If training moe models, set --fp16-no-flatten-grads to calculate correct gradnorm"

        if getattr(args, "use_tutel_moe", False):
            try:
                # To enable Tutel MoE optimizations:
                #   python3 -m pip install --user https://github.com/microsoft/tutel/releases/download/v0.1.0/tutel-0.1.0.tar.gz
                from tutel import moe as tutel_moe

                logger.info("Using microsoft Tutel plugin for fused function in MoE")
            except ModuleNotFoundError:
                raise ImportError(
                    "Please install https://github.com/microsoft/tutel/ for --use-tutel-moe"
                )

        decoder = TransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        if getattr(args, "use_stable_embedding", False):
            import bitsandbytes as bnb

            if not args.no_scale_embedding:
                logger.warning(
                    "It is recommended to pass --no-scale-embedding with --use-stable-embedding"
                )

            emb = bnb.nn.StableEmbedding(len(dictionary), embed_dim, dictionary.pad())
            # hardcode init_model_on_gpu
            if getattr(args, "init_model_on_gpu", False):
                return emb.cuda().half()
            else:
                return emb

        else:
            return Embedding(
                len(dictionary),
                embed_dim,
                dictionary.pad(),
                init_model_on_gpu=getattr(args, "init_model_on_gpu", False),
            )


def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if safe_hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if safe_hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = safe_getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = safe_getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = safe_getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.activation_fn = safe_getattr(args, "activation_fn", "relu")

    args.decoder_layerdrop = safe_getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = safe_getattr(args, "decoder_layers_to_keep", None)
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    args.base_layers = safe_getattr(args, "base_layers", 0)
    args.base_sublayers = safe_getattr(args, "base_sublayers", 1)
    args.base_shuffle = safe_getattr(args, "base_shuffle", False)

    args.add_bos_token = safe_getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = safe_getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = safe_getattr(args, "character_embeddings", False)

    args.decoder_output_dim = safe_getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = safe_getattr(
        args, "decoder_input_dim", args.decoder_embed_dim
    )

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = safe_getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = safe_getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = safe_getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = safe_getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = safe_getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = safe_getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    args.scale_fc = safe_getattr(args, "scale_fc", False)
    args.scale_attn = safe_getattr(args, "scale_attn", False)
    args.scale_heads = safe_getattr(args, "scale_heads", False)
    args.scale_resids = safe_getattr(args, "scale_resids", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.init_model_on_gpu = getattr(args, "init_model_on_gpu", False)


@register_model_architecture("transformer_lm", "transformer_lm_big")
def transformer_lm_big(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_wiki103")
@register_model_architecture("transformer_lm", "transformer_lm_baevski_wiki103")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    args.dropout = safe_getattr(args, "dropout", 0.3)
    args.adaptive_input = safe_getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = safe_getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = safe_getattr(
        args, "adaptive_input_cutoff", "20000,60000"
    )
    args.adaptive_softmax_cutoff = safe_getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = safe_getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = safe_getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = safe_getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)


@register_model_architecture("transformer_lm", "transformer_lm_gbw")
@register_model_architecture("transformer_lm", "transformer_lm_baevski_gbw")
def transformer_lm_baevski_gbw(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 512)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.no_decoder_final_norm = safe_getattr(args, "no_decoder_final_norm", True)
    transformer_lm_big(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt")
def transformer_lm_gpt(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_small")
def transformer_lm_gpt2_small(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_tiny")
def transformer_lm_gpt2_tiny(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 64)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 1)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_medium")
def transformer_lm_gpt2_medium(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1280)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 5120)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 36)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 20)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_big")
def transformer_lm_gpt2_big(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1600)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 6400)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 25)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


def base_gpt3_architecture(args):
    args.decoder_input_dim = args.decoder_embed_dim
    args.decoder_output_dim = args.decoder_embed_dim
    args.decoder_ffn_embed_dim = safe_getattr(
        args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 4
    )
    # GPT-3 used learned positional embeddings, rather than sinusoidal
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", True)
    args.dropout = safe_getattr(args, "dropout", 0.0)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.share_decoder_input_output_embed = True
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_big_wide")
def transformer_lm_gpt2_big_wide(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_bigger")
def transformer_lm_gpt2_bigger(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_layers = getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


def base_gpt3_architecture(args):
    args.decoder_input_dim = args.decoder_embed_dim
    args.decoder_output_dim = args.decoder_embed_dim
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 4
    )
    # GPT-3 used learned positional embeddings, rather than sinusoidal
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.share_decoder_input_output_embed = True
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_small")
def transformer_lm_gpt3_small(args):
    # 125M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_medium")
def transformer_lm_gpt3_medium(args):
    # 350M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_large")
def transformer_lm_gpt3_large(args):
    # 760M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1536)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_xl")
def transformer_lm_gpt3_xl(args):
    # 1.3B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_2_7")
def transformer_lm_gpt3_2_7(args):
    # 2.7B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2560)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_6_7")
def transformer_lm_gpt3_6_7(args):
    # 6.7B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 4096)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_13")
def transformer_lm_gpt3_13(args):
    # 13B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 40)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 5120)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 40)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_175")
def transformer_lm_gpt3_175(args):
    # 175B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 96)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 12288)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 96)
    base_gpt3_architecture(args)
