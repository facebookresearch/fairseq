# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)


logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('hf_gpt2')
class HuggingFaceGPT2LanguageModel(FairseqLanguageModel):

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        default_architecture(args)
        return cls(HuggingFaceGPT2Decoder(args, task))


class HuggingFaceGPT2Decoder(FairseqIncrementalDecoder):

    def __init__(self, args, task):
        super().__init__(task.target_dictionary)

        try:
            # Prepend the transformers submodule to the path, so that
            # it's prioritized over other installations. This allows
            # making local changes in the submodule.
            sys.path.insert(
                0, os.path.join(os.path.dirname(__file__), 'transformers', 'src')
            )
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            raise ImportError(
                '\n\nPlease install huggingface/transformers with:'
                '\n\n  pip install transformers'
                '\n\nOr to make local edits, install the submodule:'
                '\n\n  git submodule update --init '
                'fairseq/models/huggingface/transformers'
            )

        config = GPT2Config(
            vocab_size=len(task.target_dictionary),
            n_positions=args.max_target_positions + 1,
            n_ctx=args.max_target_positions,
            n_embd=args.embed_dim,
            n_layer=args.num_layers,
            n_head=args.num_attention_heads,
            resid_pdrop=args.dropout,
            embd_pdrop=args.dropout,
            attn_pdrop=args.attention_dropout,
            layer_norm_epsilon=1e-6,
        )
        self.model = GPT2LMHeadModel(config)

        # set zero embedding for padding symbol
        self.pad_idx = task.target_dictionary.pad()
        self.model.transformer.wte.weight.data[self.pad_idx].zero_()
        self.model.transformer.wpe.weight.data[0].zero_()

    def forward(
        self,
        prev_output_tokens,
        src_lengths=None,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
    ):
        features = self.extract_features(prev_output_tokens, incremental_state)
        lm_logits = self.model.lm_head(features)
        return (lm_logits, )

    def extract_features(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
    ):
        if incremental_state is not None:
            past = self.get_incremental_state("past")
        else:
            past = None

        # don't attend to padding symbols
        attention_mask = prev_output_tokens.ne(self.pad_idx).int()

        # set position ids to exclude padding symbols
        position_ids = attention_mask * (
            torch.arange(1, 1 + prev_output_tokens.size(1))
            .to(prev_output_tokens)
            .repeat(prev_output_tokens.size(0), 1)
        )

        outputs = self.model.transformer(
            input_ids=prev_output_tokens,
            past=past,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        last_hidden_states = outputs[0]

        if incremental_state is not None:
            self.set_incremental_state(incremental_state, "past", outputs[1])

        return last_hidden_states

    def max_positions(self):
        return self.model.config.n_positions - 1


@register_model_architecture('hf_gpt2', 'hf_gpt2')
def default_architecture(args):
    if getattr(args, 'max_target_positions', None) is None:
        args.max_target_positions = getattr(
            args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS
        )
    args.embed_dim = getattr(args, 'embed_dim', 768)
    args.num_attention_heads = getattr(args, 'num_attention_heads', 12)
    args.num_layers = getattr(args, 'num_layers', 12)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)


@register_model_architecture('hf_gpt2', 'hf_gpt2_medium')
def hf_gpt2_medium(args):
    args.embed_dim = getattr(args, 'embed_dim', 1024)
    args.num_attention_heads = getattr(args, 'num_attention_heads', 16)
    args.num_layers = getattr(args, 'num_layers', 24)
    default_architecture(args)


@register_model_architecture('hf_gpt2', 'hf_gpt2_large')
def hf_gpt2_large(args):
    args.embed_dim = getattr(args, 'embed_dim', 1280)
    args.num_attention_heads = getattr(args, 'num_attention_heads', 20)
    args.num_layers = getattr(args, 'num_layers', 36)
    default_architecture(args)


@register_model_architecture('hf_gpt2', 'hf_gpt2_xl')
def hf_gpt2_xl(args):
    args.embed_dim = getattr(args, 'embed_dim', 1600)
    args.num_attention_heads = getattr(args, 'num_attention_heads', 25)
    args.num_layers = getattr(args, 'num_layers', 48)
    default_architecture(args)
