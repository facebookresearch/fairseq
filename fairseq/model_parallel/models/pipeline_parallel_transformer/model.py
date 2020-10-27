# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.model_parallel.models.pipeline_parallel_transformer.layers import (
    Embedding,
    TransformerDecoderEmbedding,
    TransformerDecoderLayer,
    TransformerDecoderOutputLayer,
    TransformerEncoderEmbedding,
    TransformerEncoderLayer,
    TransformerEncoderLayerNorm,
)
from fairseq.models import (
    BaseFairseqModel,
    FairseqDecoder,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    base_architecture,
    transformer_iwslt_de_en,
    transformer_wmt_en_de_big,
)
from fairseq.modules import SinusoidalPositionalEmbedding


logger = logging.getLogger(__name__)


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("pipeline_parallel_transformer")
class PipelineParallelTransformerModel(BaseFairseqModel):
    def __init__(self, encoder, decoder, balance, devices, chunks, checkpoint):
        try:
            from fairscale.nn import Pipe
        except ImportError:
            raise ImportError("Please install fairscale with: pip install fairscale")
        super().__init__()
        assert isinstance(encoder, FairseqEncoder)
        assert isinstance(decoder, FairseqDecoder)
        encoder_module_list = (
            [encoder.embedding_layer]
            + list(encoder.encoder_layers)
            + [encoder.final_layer_norm]
        )
        self.num_encoder_modules = len(encoder_module_list)
        decoder_module_list = (
            [decoder.embedding_layer]
            + list(decoder.decoder_layers)
            + [decoder.decoder_output_layer]
        )
        self.num_decoder_modules = len(decoder_module_list)
        module_list = encoder_module_list + decoder_module_list
        self.devices = devices
        self.model = Pipe(
            nn.Sequential(*module_list),
            balance=balance,
            devices=devices,
            chunks=chunks,
            checkpoint=checkpoint,
        )
        self.encoder_max_positions = self.max_positions_helper(
            encoder.embedding_layer, "max_source_positions"
        )
        self.decoder_max_positions = self.max_positions_helper(
            decoder.embedding_layer, "max_target_positions"
        )
        self.adaptive_softmax = getattr(decoder, "adaptive_softmax", None)
        # Note: To be populated during inference
        self.encoder = None
        self.decoder = None

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        if self.training:
            input_lst = [src_tokens, src_lengths, prev_output_tokens]
            input = tuple(i.to(self.devices[0], non_blocking=True) for i in input_lst)
            return self.model(input)
        else:
            assert self.encoder is not None and self.decoder is not None, (
                "encoder and decoder need to be initialized by "
                + "calling the `prepare_for_inference_()` method"
            )
            encoder_output_tuple = self.encoder(input)
            return self.decoder(encoder_output_tuple)

    def prepare_for_inference_(self, cfg):
        if self.encoder is not None and self.decoder is not None:
            logger.info("Encoder and Decoder already initialized")
            return
        encoder_module_list = []
        decoder_module_list = []
        module_count = 0
        for partition in self.model.partitions:
            for module in partition:
                if module_count < self.num_encoder_modules:
                    encoder_module_list.append(module)
                else:
                    decoder_module_list.append(module)
                module_count += 1
        self.model = None
        self.encoder = TransformerEncoder(cfg.distributed_training, None, None, encoder_module_list)
        self.decoder = TransformerDecoder(
            cfg.distributed_training, None, None, decoder_module_list=decoder_module_list
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--num-embedding-chunks', type=int, metavar='N', default=1,
                            help='Number of embedding layer chunks (enables more even distribution'
                                 'of optimizer states across data parallel nodes'
                                 'when using optimizer state sharding and'
                                 'a big embedding vocabulary)')
        # fmt: on

    @classmethod
    def build_model_base(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None, num_embed_chunks=1):
            assert embed_dim % num_embed_chunks == 0, (
                f"Number of embedding chunks = {num_embed_chunks} should be "
                + f"divisible by the embedding dimension = {embed_dim}"
            )
            assert path is None or num_embed_chunks == 1, (
                "Loading embedding from a path with number of embedding chunks > 1"
                + " is not yet supported"
            )
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            # if provided, load from preloaded dictionaries
            if path:
                emb = Embedding(num_embeddings, embed_dim, padding_idx)
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            else:
                embed_chunk_dim = embed_dim // num_embed_chunks
                emb = nn.ModuleList()
                for i in range(num_embed_chunks):
                    emb.append(Embedding(num_embeddings, embed_chunk_dim, padding_idx))
            return emb

        num_embed_chunks = args.num_embedding_chunks
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = build_embedding(
                src_dict,
                args.encoder_embed_dim,
                args.encoder_embed_path,
                num_embed_chunks,
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            assert args.share_decoder_input_output_embed or num_embed_chunks == 1, (
                "Not sharing decoder I/O embeddings is not yet supported with number of "
                + "embedding chunks > 1"
            )
            encoder_embed_tokens = build_embedding(
                src_dict,
                args.encoder_embed_dim,
                args.encoder_embed_path,
                num_embed_chunks,
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict,
                args.decoder_embed_dim,
                args.decoder_embed_path,
                num_embed_chunks,
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return (encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        encoder, decoder = cls.build_model_base(args, task)
        return PipelineParallelTransformerModel(
            encoder=encoder,
            decoder=decoder,
            balance=utils.eval_str_list(args.pipeline_balance, type=int),
            devices=utils.eval_str_list(args.pipeline_devices, type=int),
            chunks=args.pipeline_chunks,
            checkpoint=args.pipeline_checkpoint,
        )

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder_max_positions, self.decoder_max_positions)

    def max_positions_helper(
        self, embedding_layer, max_positions_field="max_source_positions"
    ):
        """Maximum input length supported by the encoder or decoder."""
        if embedding_layer.embed_positions is None:
            return getattr(embedding_layer, max_positions_field)
        return min(
            getattr(embedding_layer, max_positions_field),
            embedding_layer.embed_positions.max_positions,
        )

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output, target=target)
            return out.exp_() if not log_probs else out

        # A Pipe() module returns a tuple of tensors as the output.
        # In this case, the tuple has one element - the output tensor of logits
        logits = net_output if isinstance(net_output, torch.Tensor) else net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=False)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=False)

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder_max_positions

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        is_regular_transformer = not any("model.partitions" in k for k in state_dict)
        if is_regular_transformer:
            state_dict = self.convert_to_pipeline_parallel_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict)

    def convert_to_pipeline_parallel_state_dict(self, state_dict):
        new_state_dict = self.state_dict()
        encoder_layer_idx = 0
        decoder_layer_idx = 0
        encoder_key_suffixes = [
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight",
            "self_attn_layer_norm.bias",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
            "final_layer_norm.weight",
            "final_layer_norm.bias",
        ]
        decoder_key_suffixes = [
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight",
            "self_attn_layer_norm.bias",
            "encoder_attn.k_proj.weight",
            "encoder_attn.k_proj.bias",
            "encoder_attn.v_proj.weight",
            "encoder_attn.v_proj.bias",
            "encoder_attn.q_proj.weight",
            "encoder_attn.q_proj.bias",
            "encoder_attn.out_proj.weight",
            "encoder_attn.out_proj.bias",
            "encoder_attn_layer_norm.weight",
            "encoder_attn_layer_norm.bias",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
            "final_layer_norm.weight",
            "final_layer_norm.bias",
        ]
        for pid, partition in enumerate(self.model.partitions):
            logger.info(f"Begin Partition {pid}")
            for mid, module in enumerate(partition):
                # fmt: off
                if isinstance(module, TransformerEncoderEmbedding):
                    new_state_dict[f'model.partitions.{pid}.{mid}.embed_tokens.weight'] = state_dict['encoder.embed_tokens.weight']
                    new_state_dict[f'model.partitions.{pid}.{mid}.embed_positions._float_tensor'] = state_dict['encoder.embed_positions._float_tensor']
                if isinstance(module, TransformerEncoderLayer):
                    for suffix in encoder_key_suffixes:
                        new_state_dict[f'model.partitions.{pid}.{mid}.{suffix}'] = state_dict[f'encoder.layers.{encoder_layer_idx}.{suffix}']
                    encoder_layer_idx += 1
                if isinstance(module, TransformerDecoderLayer):
                    for suffix in decoder_key_suffixes:
                        new_state_dict[f'model.partitions.{pid}.{mid}.{suffix}'] = state_dict[f'decoder.layers.{decoder_layer_idx}.{suffix}']
                    decoder_layer_idx += 1
                if isinstance(module, TransformerEncoderLayerNorm):
                    if 'encoder.layer_norm.weight' in state_dict:
                        new_state_dict[f'model.partitions.{pid}.{mid}.layer_norm.weight'] = state_dict['encoder.layer_norm.weight']
                        new_state_dict[f'model.partitions.{pid}.{mid}.layer_norm.bias'] = state_dict['encoder.layer_norm.bias']
                if isinstance(module, TransformerDecoderEmbedding):
                    new_state_dict[f'model.partitions.{pid}.{mid}.embed_tokens.weight'] = state_dict['decoder.embed_tokens.weight']
                    new_state_dict[f'model.partitions.{pid}.{mid}.embed_positions._float_tensor'] = state_dict['decoder.embed_positions._float_tensor']
                if isinstance(module, TransformerDecoderOutputLayer):
                    new_state_dict[f'model.partitions.{pid}.{mid}.output_projection.weight'] = state_dict['decoder.output_projection.weight']
                # fmt: on
        return new_state_dict


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens, encoder_module_list=None):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        try:
            from fairscale.nn import Pipe
        except ImportError:
            raise ImportError("Please install fairscale with: pip install fairscale")
        self.use_pipeline = encoder_module_list is not None
        if not self.use_pipeline:
            self.embedding_layer = TransformerEncoderEmbedding(args, embed_tokens)
            self.encoder_layers = nn.Sequential(*[TransformerEncoderLayer(args) for i in range(args.encoder_layers)])
            if isinstance(embed_tokens, nn.ModuleList):
                emb_dim = sum(e.embedding_dim for e in embed_tokens)
            else:
                emb_dim = embed_tokens.embedding_dim
            self.final_layer_norm = TransformerEncoderLayerNorm(args, emb_dim)
        else:
            encoder_balance = utils.eval_str_list(
                args.pipeline_encoder_balance, type=int
            )
            encoder_devices = utils.eval_str_list(
                args.pipeline_encoder_devices, type=int
            )
            assert sum(encoder_balance) == len(encoder_module_list), (
                f"Sum of encoder_balance={encoder_balance} is not equal "
                + f"to num_encoder_modules={len(encoder_module_list)}"
            )
            self.model = Pipe(
                module=nn.Sequential(*encoder_module_list),
                balance=encoder_balance,
                devices=encoder_devices,
                chunks=args.pipeline_chunks,
                checkpoint=args.pipeline_checkpoint,
            )

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            input_tuple(
                src_tokens (LongTensor): tokens in the source language of shape
                    `(batch, src_len)`
                src_lengths (torch.LongTensor): lengths of each source sentence of
                    shape `(batch)`
            )

        Returns:
            output_tuple(
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - prev_output_tokens
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
            )
        """
        dummy_prev_output_tokens = torch.zeros(
            1, dtype=src_tokens.dtype, device=src_tokens.device
        )
        input_tuple = (src_tokens, src_lengths, dummy_prev_output_tokens)
        if self.use_pipeline:
            input_tuple = tuple(i.to(self.model.devices[0]) for i in input_tuple)
            encoder_out = self.model(input_tuple)
        else:
            encoder_embed_output_tuple = self.embedding_layer(input_tuple)
            encoder_layers_output = self.encoder_layers(encoder_embed_output_tuple)
            encoder_out = self.final_layer_norm(encoder_layers_output)
        # first element is the encoder output
        # second element is the encoder padding mask
        # the remaining elements of EncoderOut are not computed by
        # the PipelineParallelTransformer
        return EncoderOut(encoder_out[0], encoder_out[1], None, None, None, None)

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embedding_layer.embed_positions is None:
            return self.embedding_layer.max_source_positions
        return min(
            self.embedding_layer.max_source_positions,
            self.embedding_layer.embed_positions.max_positions,
        )


class TransformerDecoder(FairseqDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
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
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        decoder_module_list=None,
    ):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        try:
            from fairscale.nn import Pipe
        except ImportError:
            raise ImportError("Please install fairscale with: pip install fairscale")
        self.use_pipeline = decoder_module_list is not None
        if not self.use_pipeline:
            self.embedding_layer = TransformerDecoderEmbedding(args, embed_tokens)
            self.decoder_layers = nn.Sequential(*[
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ])
            self.decoder_output_layer = TransformerDecoderOutputLayer(
                args, embed_tokens, dictionary
            )
        else:
            decoder_balance = utils.eval_str_list(
                args.pipeline_decoder_balance, type=int
            )
            decoder_devices = utils.eval_str_list(
                args.pipeline_decoder_devices, type=int
            )
            assert sum(decoder_balance) == len(decoder_module_list), (
                f"Sum of decoder_balance={decoder_balance} is not equal "
                + f"to num_decoder_modules={len(decoder_module_list)}"
            )
            self.model = Pipe(
                module=nn.Sequential(*decoder_module_list),
                balance=decoder_balance,
                devices=decoder_devices,
                chunks=args.pipeline_chunks,
                checkpoint=args.pipeline_checkpoint,
            )

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        input_tuple = (
            encoder_out.encoder_out,
            encoder_out.encoder_padding_mask,
            prev_output_tokens,
        )
        if self.use_pipeline:
            input_tuple = tuple(i.to(self.model.devices[0]) for i in input_tuple)
            return (self.model(input_tuple),)
        else:
            embed_layer_output = self.embedding_layer(input_tuple)
            state = self.decoder_layers(embed_layer_output)
            return (self.decoder_output_layer(state),)

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embedding_layer.embed_positions is None:
            return self.embedding_layer.max_target_positions
        return min(
            self.embedding_layer.max_target_positions,
            self.embedding_layer.embed_positions.max_positions,
        )

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


@register_model_architecture(
    "pipeline_parallel_transformer", "transformer_iwslt_de_en_pipeline_parallel"
)
def transformer_iwslt_de_en_dist(args):
    transformer_iwslt_de_en(args)


@register_model_architecture(
    "pipeline_parallel_transformer", "transformer_wmt_en_de_big_pipeline_parallel"
)
def transformer_wmt_en_de_big_dist(args):
    transformer_wmt_en_de_big(args)
