
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

import torch.nn as nn  # mypy: ignore

from .t5_encoder import T5Encoder
from .t5_decoder import T5Decoder


@register_model('t5')
class T5Model(FairseqEncoderDecoderModel):
    """
    T5 model from `"Exploring the Limits of Transfer Learning with a Unified
    Text-to-Text Transformer" (Raffel, 2019) <https://arxiv.org/abs/1910.10683>`.

    Args:
        encoder: the encoder
        decoder: the decoder
        shared_emb: shared word embeddings

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
    :ref: fairseq.models.t5.t5model.add_args
    :prog:

    """
    def __init__(self, args, encoder: T5Encoder, decoder: T5Decoder, shared_emb: nn.Embedding):
        super().__init__(encoder, decoder)
        self.shared = shared_emb
        self.args = args
        self.supports_align_args = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            '--d-ff', type=int, metavar='N',
            help='embedding dimention in FFN')
        parser.add_argument(
            '--d-kv', type=int, metavar='N',
            help='embedding dimention of K, V in attention')
        parser.add_argument(
            '--d-model', type=int, metavar='N',
            help='dimention of hidden state')
        parser.add_argument(
            '--dropout-rate', type=int, metavar='N',
            help='embedding dimention in FFN')
        parser.add_argument(
            '--layer-norm-epsilon', type=int, metavar='N',
            help='epsilon to use in layer normalization layers')
        parser.add_argument(
            '--n-positions', type=int, metavar='N',
            help='')
        parser.add_argument(
            '--num-heads', type=int, metavar='N',
            help='number of attention heads')
        parser.add_argument(
            '--num-layers', type=int, metavar='N',
            help='number of layers in encoder (decoder)')
        parser.add_argument(
            '--relative-attention-num-buckets', type=int, metavar='N',
            help='number of backets in relative attention')
        parser.add_argument(
            '--vocab-size', type=int, metavar='N',
            help='vocabulary size (number of words and special tokens)')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        shared_emb = nn.Embedding(args.vocab_size, args.d_model)
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder = cls.build_encoder(args, src_dict, shared_emb)
        decoder = cls.build_decoder(args, tgt_dict, shared_emb)
        model = cls(args, encoder, decoder, shared_emb)
        return model

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return T5Encoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return T5Decoder(args, tgt_dict, embed_tokens, False)

    def param_groups(self):
        matrix_params = list(filter(lambda p: p.requires_grad and p.dim() == 2,
                                    self.parameters()))
        vector_params = list(filter(lambda p: p.requires_grad and p.dim() == 1,
                                    self.parameters()))
        params = [{'params': matrix_params}, {'params': vector_params}]
        return params


@register_model_architecture('t5', 't5-small')
def t5_small_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 2048)
    args.d_kv = getattr(args, 'd_kv', 64)
    args.d_model = getattr(args, 'd_model', 512)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 8)
    args.num_layers = getattr(args, 'num_layers', 6)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.vocab_size = getattr(args, 'vocab_size', 32128)


@register_model_architecture('t5', 't5-base')
def t5_base_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 3072)
    args.d_kv = getattr(args, 'd_kv', 64)
    args.d_model = getattr(args, 'd_model', 768)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 12)
    args.num_layers = getattr(args, 'num_layers', 12)
    args.vocab_size = getattr(args, 'vocab_size', 32128)


@register_model_architecture('t5', 't5-large')
def t5_large_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 4096)
    args.d_kv = getattr(args, 'd_kv', 64)
    args.d_model = getattr(args, 'd_model', 1024)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 16)
    args.num_layers = getattr(args, 'num_layers', 24)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.vocab_size = getattr(args, 'vocab_size', 32128)


@register_model_architecture('t5', 't5-3b')
def t5_3b_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 16384)
    args.d_kv = getattr(args, 'd_kv', 128)
    args.d_model = getattr(args, 'd_model', 1024)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 32)
    args.num_layers = getattr(args, 'num_layers', 24)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.vocab_size = getattr(args, 'vocab_size', 32128)


@register_model_architecture('t5', 't5-11b')
def t5_11b_architecture(args):
    args.d_ff = getattr(args, 'd_ff', 65536)
    args.d_kv = getattr(args, 'd_kv', 128)
    args.d_model = getattr(args, 'd_model', 1024)
    args.dropout_rate = getattr(args, 'dropout_rate', 0.1)
    args.layer_norm_epsilon = getattr(args, 'layer_norm_epsilon', 1e-06)
    args.n_positions = getattr(args, 'n_positions', 512)
    args.num_heads = getattr(args, 'num_heads', 128)
    args.num_layers = getattr(args, 'num_layers', 24)
    args.relative_attention_num_buckets = getattr(args, 'relative_attention_num_buckets', 32)
    args.vocab_size = getattr(args, 'vocab_size', 32128)
