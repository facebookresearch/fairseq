# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
from dataclasses import dataclass, field, fields
from typing import List, Optional

from omegaconf import II

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.utils import safe_getattr, safe_hasattr

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

_NAME_PARSER = r"(decoder|encoder|quant_noise)_(.*)"


@dataclass
class EncDecBaseConfig(FairseqDataclass):
    embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained embedding"}
    )
    embed_dim: Optional[int] = field(
        default=512, metadata={"help": "embedding dimension"}
    )
    ffn_embed_dim: int = field(
        default=2048, metadata={"help": "embedding dimension for FFN"}
    )
    layers: int = field(default=6, metadata={"help": "number of layers"})
    attention_heads: int = field(
        default=8, metadata={"help": "number of attention heads"}
    )
    normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each block"}
    )
    learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings"}
    )
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    layerdrop: float = field(default=0, metadata={"help": "LayerDrop probability"})
    layers_to_keep: Optional[List[int]] = field(
        default=None, metadata={"help": "which layers to *keep* when pruning"}
    )


@dataclass
class DecoderConfig(EncDecBaseConfig):
    input_dim: int = II("model.decoder.embed_dim")
    output_dim: int = field(
        default=II("model.decoder.embed_dim"),
        metadata={
            "help": "decoder output dimension (extra linear layer if different from decoder embed dim)"
        },
    )

    def __post_init__(self):
        #  II doesn't work if we are just creating the object outside of hydra so fix that
        if self.input_dim == II("model.decoder.embed_dim"):
            self.input_dim = self.embed_dim
        if self.output_dim == II("model.decoder.embed_dim"):
            self.output_dim = self.embed_dim


@dataclass
class QuantNoiseConfig(FairseqDataclass):
    pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )


@dataclass
class TransformerConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu",
        metadata={"help": "activation function to use"},
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN.",
            "alias": "--relu-dropout",
        },
    )
    adaptive_input: bool = False
    encoder: EncDecBaseConfig = EncDecBaseConfig()
    # TODO should really be in the encoder config
    max_source_positions: int = field(
        default=DEFAULT_MAX_SOURCE_POSITIONS,
        metadata={"help": "Maximum input length supported by the encoder"},
    )
    decoder: DecoderConfig = DecoderConfig()
    # TODO should really be in the decoder config
    max_target_positions: int = field(
        default=DEFAULT_MAX_TARGET_POSITIONS,
        metadata={"help": "Maximum output length supported by the decoder"},
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    share_all_embeddings: bool = field(
        default=False,
        metadata={
            "help": "share encoder, decoder and output embeddings (requires shared dictionary and embed dim)"
        },
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if True, disables positional embeddings (outside self attention)"
        },
    )
    adaptive_softmax_cutoff: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0.0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
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
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute"
        },
    )
    offload_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations."
        },
    )
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    no_cross_attention: bool = field(
        default=False, metadata={"help": "do not perform cross-attention"}
    )
    cross_self_attention: bool = field(
        default=False, metadata={"help": "perform cross+self-attention"}
    )
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise: QuantNoiseConfig = field(default=QuantNoiseConfig())
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )
    # DEPRECATED field, but some old checkpoints might have it
    char_inputs: bool = field(
        default=False, metadata={"help": "if set, model takes character ids as input"}
    )
    relu_dropout: float = 0.0
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

    export: bool = field(
        default=False,
        metadata={"help": "make the layernorm exportable with torchscript."},
    )

    # copied from transformer_lm but expected in transformer_decoder:
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )

    # We need to make this hierarchical dataclass like the flat namespace
    # __getattr__ and __setattr__ here allow backward compatibility
    # for subclasses of Transformer(Legacy) that depend on read/write on
    # the flat namespace.

    def __getattr__(self, name):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            return safe_getattr(sub, match[2])
        raise AttributeError(f"invalid argument {name}.")

    def __setattr__(self, name, value):
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            setattr(sub, match[2], value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def _copy_keys(args, cls, prefix, seen):
        """
        copy the prefixed keys (decoder_embed_dim) to the DC fields: decoder.embed_dim
        """
        cfg = cls()
        for fld in fields(cls):
            # for all the fields in the DC, find the fields (e.g. embed_dim)
            # in the namespace with the prefix (e.g. decoder)
            # and set it on the dc.
            args_key = f"{prefix}_{fld.name}"
            if safe_hasattr(args, args_key):
                seen.add(args_key)
                setattr(cfg, fld.name, safe_getattr(args, args_key))
            if safe_hasattr(args, fld.name):
                seen.add(fld.name)
                setattr(cfg, fld.name, safe_getattr(args, fld.name))
        return cfg

    @classmethod
    def from_namespace(cls, args):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = DecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, DecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = EncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, EncDecBaseConfig, "encoder", seen
                        )
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                elif safe_hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args
