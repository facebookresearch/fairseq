# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from fairseq.models import transformer
from fairseq.tasks import online_backtranslation
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional
import argparse
from fairseq import options
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from torch import Tensor
from fairseq.models.transformer import TransformerModel, base_architecture

"""
Convert XLM name to fairseq name

xlm_model['model'/'encoder'] -> fairseq_model['model']

key: 'module.' -> 'encoder.'

"""

import logging

logger = logging.getLogger(__name__)

N_MAX_POSITIONS = 512  # maximum input sequence length

DECODER_ONLY_PARAMS = [
    'layer_norm15.%i.weight', 'layer_norm15.%i.bias',
    'encoder_attn.%i.q_lin.weight', 'encoder_attn.%i.q_lin.bias',
    'encoder_attn.%i.k_lin.weight', 'encoder_attn.%i.k_lin.bias',
    'encoder_attn.%i.v_lin.weight', 'encoder_attn.%i.v_lin.bias',
    'encoder_attn.%i.out_lin.weight', 'encoder_attn.%i.out_lin.bias'
]

TRANSFORMER_LAYER_PARAMS = [
    'attentions.%i.q_lin.weight', 'attentions.%i.q_lin.bias',
    'attentions.%i.k_lin.weight', 'attentions.%i.k_lin.bias',
    'attentions.%i.v_lin.weight', 'attentions.%i.v_lin.bias',
    'attentions.%i.out_lin.weight', 'attentions.%i.out_lin.bias',
    'layer_norm1.%i.weight', 'layer_norm1.%i.bias',
    'ffns.%i.lin1.weight', 'ffns.%i.lin1.bias',
    'ffns.%i.lin2.weight', 'ffns.%i.lin2.bias',
    'layer_norm2.%i.weight', 'layer_norm2.%i.bias'
]


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    # nn.init.normal_(m.weight, mean=0, std=1)
    # nn.init.xavier_uniform_(m.weight)
    # nn.init.constant_(m.bias, 0.)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [float(pos) / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, params):
        super().__init__()
        self.asm = params.asm
        self.label_smoothing = getattr(params, 'label_smoothing', 0)
        self.unlikelihood_alpha = getattr(params, 'unlikelihood_alpha', 1.0)
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim

        if params.asm is False:
            self.proj = Linear(dim, params.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=params.n_words,
                cutoffs=params.asm_cutoffs,
                div_value=params.asm_div_value,
                head_bias=True,  # default is False
            )

    def _label_smoothed_nll_loss(self, lprobs, targets):
        if targets.dim() == lprobs.dim() - 1:
            targets = targets.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=targets)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        non_pad_mask = targets.ne(self.pad_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = self.label_smoothing / lprobs.size(-1)
        loss = (1. - self.label_smoothing) * nll_loss + eps_i * smooth_loss
        return loss

    def forward(self, x, y, get_scores=False, unlikelihood=None):
        """
        Compute the loss, and optionally the scores.
        """
        assert (y == self.pad_index).sum().item() == 0

        if self.asm is False:
            scores = self.proj(x).view(-1, self.n_words)
            if self.label_smoothing > 0:
                loss = self._label_smoothed_nll_loss(F.log_softmax(scores, dim=-1), y)
            else:
                loss = F.cross_entropy(scores, y, reduction='mean')
        else:
            _, loss = self.proj(x, y)
            scores = self.proj.log_prob(x) if get_scores else None

        if unlikelihood is not None and self.unlikelihood_alpha > 0:
            unlikelihood = unlikelihood.long()
            # torch.clamp((1.0 - lprobs.exp()), min=1e-5)
            neg_softmax = torch.clamp((1.0 - F.softmax(scores, dim=-1)), min=1e-5)
            neg_softmax_c = torch.index_select(neg_softmax, 1, unlikelihood)
            unlike_loss = -neg_softmax_c.log().sum(dim=-1).mean()
            loss += unlike_loss * self.unlikelihood_alpha

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)
    
    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation).
        Follow fairseq.modules.multihead_attention.MultiHeadAttention
        """
        
        cache = incremental_state
        if self.layer_id in cache:
            items = cache[self.layer_id]
            type_ = type(items)
            cache[self.layer_id] = type_([x.index_select(0, new_order) for x in items])

        return incremental_state

    def forward(self, input, mask, kv=None, cache=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))                                          # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))                                          # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))                                          # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)                             # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)                             # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
                    k = k[:, :, :klen]
                    v = v[:, :, :klen]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)                                       # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))                           # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)               # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float('inf'))  # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)           # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)                                    # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)                                            # (bs, qlen, dim)

        return self.out_lin(context)


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_dim, dim_hidden)
        self.lin2 = Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else F.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# --------- Actual Fairseq Transformer for XLM

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def add_xlmcode_args(parser):
    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--share_enc", type=int, default=-1,
                        help="Number of Transformer layers")
    parser.add_argument("--share_dec", type=int, default=-1,
                        help="Number of Transformer layers")
    parser.add_argument("--n_dec_layers", type=int, default=6,
                        help="Number of Decoder Transformer layers")

    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--use_lang_emb", type=bool_flag, default=True,
                        help="Use language embedding")

    parser.add_argument("--bertnmt_dropnet", type=float, default=1.0,
                        help="bertnmt_dropnet")
    parser.add_argument("--pos_embed_last", type=bool_flag, default=True,
                        help="Deberta use embed last")
    parser.add_argument("--relative_attention", type=bool_flag, default=True,
                        help="relative_attention")
    parser.add_argument("--max_relative_positions", type=int, default=-1,
                        help="max_relative_positions")
    parser.add_argument("--position_buckets", type=int, default=-1,
                        help="position_buckets")

    # memory parameters
    parser.add_argument("--use_memory", type=bool_flag, default=False,
                        help="Use an external memory")

    # adaptive softmax
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")


@register_model('xlm_transformer_encoder')
class XLMCodeTransformerFairseqEncoderModel(FairseqEncoderModel):

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        
        # We follow BERT's random weight initialization
        # self.apply(init_bert_params)

    @staticmethod
    def add_args(parser):
        add_xlmcode_args(parser)
    
    @classmethod
    def build_model(cls, args, task):
        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, "max_positions"):
            args.max_positions = getattr(args, "tokens_per_sample", getattr(args, "max_source_positions", None))

        encoder = XLMCodeTransformerEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def max_positions(self):
        return N_MAX_POSITIONS

    def forward(
        self,
        src_tokens,
        src_lengths,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        src_langs = kwargs.get("src_langs", None)
        kwargs.pop("src_langs", None)
        assert src_langs is not None
        x = self.encoder(
            src_tokens, src_lengths=src_lengths,
            features_only=features_only, return_all_hiddens=return_all_hiddens, 
            src_langs=src_langs,
            in_model_forward=True,
            **kwargs
        )
        fake_extra = {'x': x}
        return x, fake_extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    @property
    def supported_targets(self):
        return {"self"}


@register_model_architecture("xlm_transformer_encoder", "xlm_transformer_encoder")
def xlm_base_architecture(args):
    args.emb_dim = getattr(args, "emb_dim", 512)
    args.n_layers = getattr(args, "n_layers", 6)
    args.n_heads = getattr(args, "n_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.gelu_activation = getattr(args, "gelu_activation", True)
    transformer.base_architecture(args)


@register_model_architecture("xlm_transformer_encoder", "xlm_transformer_encoder_big")
def xlm_transformer_encoder_big(args):
    args.emb_dim = getattr(args, "emb_dim", 1024)
    args.n_layers = getattr(args, "n_layers", 6)
    args.n_heads = getattr(args, "n_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.gelu_activation = getattr(args, "gelu_activation", True)
    xlm_base_architecture(args)


@register_model('xlm_transformer')
class XLMCodeTransformerFairseqModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.args = args
        args.left_pad_source = options.eval_bool(getattr(args, "left_pad_source", "False"))
        args.left_pad_target = options.eval_bool(getattr(args, "left_pad_target", "False"))
        assert not args.left_pad_source, f'For XLM left_pad_source must be false: {type(args.left_pad_source)}'
        assert not args.left_pad_target, f'For XLM left_pad_target must be false: {type(args.left_pad_target)}'
        self.supports_align_args = True
        self.enc_dictionary = encoder.dictionary
        self.dec_dictionary = decoder.dictionary
        self.bos_index = self.enc_dictionary.bos()
        self.eos_index = self.enc_dictionary.eos()
        self.pad_index = self.enc_dictionary.pad()
        self.mono_langs = getattr(self.args, "mono_langs", None)
        self.strictly_mt = getattr(self.args, "strictly_mt", False)
        self.monolangs2srclangs_map = None

    def maybe_build_mono2langs_map(self, dictionary=None):
        dictionary = dictionary or self.enc_dictionary
        try:
            self.encoder.maybe_build_mono2langs_map(dictionary=dictionary)
        except Exception as e:
            logger.warning(f'failed to build encoder.maybe_build_mono2langs_map')
            raise e
        try:
            self.decoder.maybe_build_mono2langs_map(dictionary=dictionary)
        except Exception as e:
            logger.warning(f'failed to build decoder.maybe_build_mono2langs_map')
            raise e
        
        if self.monolangs2srclangs_map is not None:
            return
        if self.mono_langs is not None:
            self.mono_langs = self.mono_langs.split(",") if isinstance(self.mono_langs, str) else self.mono_langs
            assert isinstance(self.mono_langs, list), f'{self.mono_langs}'
            assert len(self.mono_langs) == self.encoder.n_langs
            self.monolangs2srclangs_map = {
                online_backtranslation._lang_token_index(dictionary, lang): i
                for i, lang in enumerate(self.mono_langs)
            }
            logger.info(
                f'{self.__class__.__name__} found mono_langs, expected OnlineBackTranslationTask: '
                f'{self.mono_langs}: {self.monolangs2srclangs_map}'
            )

    @staticmethod
    def add_args(parser):
        add_xlmcode_args(parser)
    
    @classmethod
    def encoder_with_output(cls):
        return False
    
    @classmethod
    def build_model(cls, args, task):
        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, "max_positions"):
            args.max_positions = getattr(args, "tokens_per_sample", args.max_source_positions)
        encoder = XLMCodeTransformerEncoder(args, task.source_dictionary, with_output=cls.encoder_with_output())
        decoder = XLMCodeTransformerDecoder(args, task.target_dictionary)
        return cls(args, encoder, decoder)

    def resolve_xlm_encoder_state_dict(self, state_dict):
        # resolve of state_dict is just the pretrained encoder
        # for k, v in state_dict.items():
        state_keys = list(state_dict.keys())
        for k in state_keys:
            if k.startswith("encoder."):
                # NOTE: only replace if decoder. not exists in state_dict
                dec_key = k.replace("encoder.", "decoder.")
                if dec_key not in state_dict:
                    state_dict[dec_key] = state_dict[k]

        for k, v in self.state_dict().items():
            # reassign cross-attention
            if k not in state_dict:
                assert (
                    k.startswith('decoder.layer_norm15') or 
                    k.startswith("decoder.encoder_attn")), f'invalid key: {k} missing in state_dict'
                state_dict[k] = v

    def resolve_with_without_output(self, state_dict):
        if not self.encoder.with_output:
            if "encoder.pred_layer.proj.weight" in state_dict:
                state_dict.pop("encoder.pred_layer.proj.weight", None)
            if "encoder.pred_layer.proj.bias" in state_dict:
                state_dict.pop("encoder.pred_layer.proj.bias", None)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        self.resolve_xlm_encoder_state_dict(state_dict)
        self.resolve_with_without_output(state_dict)
        
        # detect and expand dictionary to cover extra lang-specifier, by copying
        if self.mono_langs is not None:
            def extend_embed(state_key, m_weight):
                if state_key not in self.state_dict() or state_key not in state_dict:
                    return
                state_weight = state_dict[state_key]
                if m_weight.shape[0] != state_weight.shape[0]:
                    assert m_weight.size(0) > state_weight.size(0)
                    logger.info(
                        f'{self.__class__.__name__}: extend state_dict {state_key}: '
                        f'{state_weight.shape} -> {m_weight.shape}'
                    )
                    state_dict[state_key] = torch.cat((
                        state_weight,
                        m_weight[state_weight.size(0):].to(state_weight.device)
                    ), 0)
            extend_embed("encoder.embeddings.weight", self.encoder.embeddings.weight)
            if self.encoder.with_output:
                extend_embed("encoder.pred_layer.proj.weight", self.encoder.pred_layer.proj.weight)
                extend_embed("encoder.pred_layer.proj.bias", self.encoder.pred_layer.proj.bias)
            extend_embed("decoder.embeddings.weight", self.decoder.embeddings.weight)
            extend_embed("decoder.pred_layer.proj.weight", self.decoder.pred_layer.proj.weight)
            extend_embed("decoder.pred_layer.proj.bias", self.decoder.pred_layer.proj.bias)

    def infer_langs(self, tokens, key, kwargs, backup_id):
        if tokens is None:
            return None
        langs_key = f'{key}_langs'
        lang_id_key = f'{key}_lang_id'
        if langs_key in kwargs:
            langs = kwargs[langs_key]
        elif lang_id_key in kwargs:
            langs = kwargs[lang_id_key]
        elif not torch.any((tokens[:, 0] == self.eos_index) | (tokens[:, 0] == self.bos_index)):
            langs = torch.tensor(
                [self.monolangs2srclangs_map[x.item()] for i, x in enumerate(tokens[:, 0])]
            ).type_as(tokens).to(tokens.device)
        else:
            # langs = tokens.new(tokens.size(0)).fill_(backup_id)
            raise ValueError(f'backup invalid: {backup_id}')
        return langs

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        **Must self-infer src_langs and tgt_langs from the input of various tasks:
        0. If src_langs, tgt_langs in kwargs -> trivial
        1. TranslationTask
            1.1 if src_lang_id / tgt_lang_id exists
                build src_langs from src_lang_id
            1.2 else, FIXME nxphi: this case could be of wrong impl, please avoid falling into
                cast src_langs=0, tgt_langs=1. This features is temporarily turned off
        2. MLM training
            2.1 MultiLingualMaskedLMXLMTask must specify src_langs
        3. OnlineBackTranslation: input always contain __${lang}__ as bos
            3.0 pass mono_langs into the model
            3.1 extract __lang__ bos to build src_langs / tgt_langs
            3.2 replace bos with eos

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if alignment_layer is not None or alignment_heads is not None:
            raise NotImplementedError('alignment not impl.')
        
        # infer src_langs
        src_langs = self.infer_langs(src_tokens, "src", kwargs, 0)
        tgt_langs = self.infer_langs(prev_output_tokens, "tgt", kwargs, 1)
        assert src_langs is not None
        kwargs.pop("src_langs", None)
        kwargs.pop("tgt_langs", None)

        x = self.encoder(
            src_tokens, src_lengths, 
            features_only=True, 
            return_all_hiddens=True, 
            src_langs=src_langs, 
            in_model_forward=True, **kwargs
        )
        if features_only:
            return x, {}

        assert tgt_langs is not None
        src_enc = x
        src_len = src_lengths
        tgt_len = kwargs.get("tgt_lengths", (prev_output_tokens != self.decoder.pad_index).int().sum(-1))

        dec_out = self.decoder(
            prev_output_tokens, tgt_len,
            src_enc=src_enc, src_len=src_len, src_langs=tgt_langs, features_only=False,
            in_model_forward=True
        )
        return dec_out


@register_model_architecture("xlm_transformer", "xlm_transformer_big")
def xlm_transformer_big(args):
    args.emb_dim = getattr(args, "emb_dim", 1024)
    args.n_layers = getattr(args, "n_layers", 6)
    args.n_heads = getattr(args, "n_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.gelu_activation = getattr(args, "gelu_activation", True)
    base_architecture(args)


class _XLMCodeTransformerEncDecWrapper(object):
    """
    Transformer Encoder/Decoder adapted from XLM codebase
    """
    def __init__(self, args, dictionary, with_output=True):
        super().__init__(dictionary)
        # self.is_encoder = True
        # self.is_decoder = False
        self.with_output = with_output

        # FIXME nxphi: temporary fix n_langs=2 for mass en-ro
        self.n_langs = getattr(args, "n_langs", 2)
        self.args = args
        self.mono_langs = getattr(args, "mono_langs", None)
        self.n_words = len(dictionary)
        self.eos_index = dictionary.eos_index
        self.pad_index = dictionary.pad_index
        self.bos_index = dictionary.bos_index

        args.n_langs = self.n_langs
        args.n_words = self.n_words
        args.eos_index = self.eos_index
        args.pad_index = self.pad_index

        self.mono_langs = getattr(args, "mono_langs", None)
        self.monolangs2srclangs_map = None
        self.mono_langs_ids = None
        self.use_lang_emb = getattr(args, 'use_lang_emb', True)

        # model parameters
        self.dim = args.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = args.n_heads  # 8 by default
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.attention_dropout = args.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'
        self.max_source_positions = getattr(args, 'max_source_positions', N_MAX_POSITIONS)
        self.max_target_positions = getattr(args, 'max_target_positions', N_MAX_POSITIONS)

        # embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if args.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        if self.n_langs > 1 and self.use_lang_emb:
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        self.memories = nn.ModuleDict()
        if getattr(args, 'use_memory', False):
            raise NotImplementedError('use_memory true not impl')

        for layer_id in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            if ('%i_in' % layer_id) in self.memories:
                self.ffns.append(None)
            else:
                self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout, gelu_activation=args.gelu_activation))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        # output layer
        if self.with_output:
            self.pred_layer = PredLayer(args)
            if args.share_inout_emb:
                self.pred_layer.proj.weight = self.embeddings.weight

    def maybe_build_mono2langs_map(self, dictionary=None):
        """
        mono2langs help infer language of sentences from the lang_id start token
            e.g: mono2langs convert __en__ -> index 0
        UMT model from this class should:
            (0) infer language from starting token __langid__
            (1) replace __langid__ into </s>
            (2) apply language embedding accordingly
        """
        dictionary = dictionary or self.dictionary
        if self.monolangs2srclangs_map is not None:
            return
        if self.mono_langs is not None:
            self.mono_langs = self.mono_langs.split(",") if isinstance(self.mono_langs, str) else self.mono_langs
            assert isinstance(self.mono_langs, list), f'{self.mono_langs}'
            assert len(self.mono_langs) == self.n_langs, f'{self.mono_langs=}, {self.n_langs=}'
            self.monolangs2srclangs_map = {
                online_backtranslation._lang_token_index(dictionary, lang): i
                for i, lang in enumerate(self.mono_langs)
            }
            self.mono_langs_ids = list(self.monolangs2srclangs_map.keys())
            logger.info(
                f'{self.__class__.__name__} found mono_langs, expected OnlineBackTranslationTask: '
                f'{self.mono_langs}: {self.monolangs2srclangs_map}, {self.mono_langs_ids=}'
            )

    def infer_langs(self, tokens, backup_id):
        if not torch.any((tokens[:, 0] == self.eos_index) | (tokens[:, 0] == self.bos_index)):
            langs = torch.tensor(
                [self.monolangs2srclangs_map[x.item()] for i, x in enumerate(tokens[:, 0])]
            ).type_as(tokens).to(tokens.device)
        else:
            raise ValueError(f'invalid case: {tokens[:, 0]=}')
        return langs

    def compute_output(self, tensor, masked_tokens=None):
        if self.with_output:
            _features = tensor
            if masked_tokens is not None:
                _features = _features[masked_tokens, :]
            tensor = self.pred_layer.proj(_features)
        return tensor

    def add_dummy_to_output(self, tensor):
        if hasattr(self, "pred_layer"):
            return self.pred_layer.proj(tensor.new(1, tensor.size(-1)).fill_(0))[0, 0] * 0
        return 0.0


class XLMCodeTransformerEncoder(_XLMCodeTransformerEncDecWrapper, FairseqEncoder):
    """
    Actual XLM code transformer Encoder
    """
    def __init__(self, args, dictionary, with_output=True):
        self.is_encoder = True
        self.is_decoder = False
        super().__init__(args, dictionary, with_output)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.position_embeddings is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.position_embeddings.num_embeddings)

    def forward(
        self, 
        src_tokens, 
        src_lengths,
        in_model_forward=False,
        features_only=False, 
        return_all_hiddens=True, 
        masked_tokens=None, 
        causal=False, 
        src_enc=None, 
        src_len=None, 
        positions=None, 
        src_langs=None, 
        cache=None, 
        enc_mask=None,
        **kwargs
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        if not in_model_forward:
            src_langs = self.infer_langs(src_tokens, 0) if src_langs is None else src_langs
        
        src_tokens_cl = src_tokens
        if self.mono_langs is not None:
            src_tokens_cl = src_tokens.clone()
            src_tokens_cl[:, 0] = self.eos_index
            
        # check inputs
        causal = False
        x = src_tokens_cl.transpose(0, 1)
        lengths = src_lengths
        langs = src_langs.unsqueeze(0).expand_as(x)
        
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]

        # embeddings
        tensor = self.embeddings(x)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            # memory
            if ('%i_after' % i) in self.memories:
                tensor = tensor + self.memories['%i_after' % i](tensor)
            # TODO: add extra layer norm here?

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # # move back sequence length to dimension 0 DO NOT TRANSPOSE BACK
        # tensor = tensor.transpose(0, 1)

        # dummy pred_layer to prevent multi-output problem
        tensor += self.add_dummy_to_output(tensor)

        if not features_only:
            tensor = self.compute_output(tensor, masked_tokens)

        # return tensor
        if in_model_forward:
            return tensor
        else:
            return {
                "src_enc": [tensor],
                "src_len": [lengths],
                "src_langs": [src_langs],
            }
    
    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["src_enc"]) == 0:
            new_src_enc = []
        else:
            new_src_enc = [encoder_out["src_enc"][0].index_select(0, new_order)]
        
        if len(encoder_out["src_len"]) == 0:
            new_src_len = []
        else:
            new_src_len = [encoder_out["src_len"][0].index_select(0, new_order)]
        
        if len(encoder_out["src_langs"]) == 0:
            new_src_langs = []
        else:
            new_src_langs = [encoder_out["src_langs"][0].index_select(0, new_order)]

        return {
            "src_enc": new_src_enc,
            "src_len": new_src_len,
            "src_langs": new_src_langs,
        }


class XLMCodeTransformerDecoder(_XLMCodeTransformerEncDecWrapper, FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, with_output=True):
        self.is_encoder = False
        self.is_decoder = True
        super().__init__(args, dictionary, with_output)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.position_embeddings is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.position_embeddings.num_embeddings)

    def forward(
        self, 
        src_tokens,
        src_lengths=None,
        in_model_forward=False,
        features_only=False,
        return_all_hiddens=True, 
        masked_tokens=None, 
        causal=True, 
        encoder_out=None,
        src_enc=None,
        src_len=None, 
        positions=None, 
        src_langs=None, 
        cache=None, 
        enc_mask=None,
        incremental_state=None,
        **kwargs
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        if not in_model_forward:
            assert encoder_out is not None
            src_enc = encoder_out['src_enc'][0]
            src_len = encoder_out['src_len'][0]
            src_langs = self.infer_langs(src_tokens, 1) if src_langs is None else src_langs
            cache = incremental_state
            if "slen" not in cache:
                cache['slen'] = 0
        
        src_tokens_cl = src_tokens
        if self.mono_langs is not None:
            src_tokens_cl = src_tokens.clone()
            src_tokens_cl[:, 0] = self.eos_index
            
        # check inputs
        assert not features_only
        causal = True
        x = src_tokens_cl.transpose(0, 1)
        lengths = src_lengths if src_lengths is not None else (src_tokens_cl != self.pad_index).int().sum(-1)
        if src_langs is not None:
            langs = src_langs.unsqueeze(0).expand_as(x)
            # unique_lang_ids = torch.unique(src_langs, sorted=True)
        else:
            langs = None
        
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
            src_enc = src_enc[:, :src_len.max()]

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]
            if enc_mask is not None:
                src_mask &= enc_mask
            assert src_mask.size(-1) == src_enc.size(1)

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        tensor = self.embeddings(x)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            # memory
            if ('%i_after' % i) in self.memories:
                tensor = tensor + self.memories['%i_after' % i](tensor)
            # TODO: add extra layer norm here?

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # # move back sequence length to dimension 0 DO NOT TRANSPOSE BACK
        # tensor = tensor.transpose(0, 1)

        # dummy pred_layer to prevent multi-output problem
        tensor += self.add_dummy_to_output(tensor)

        if not features_only:
            tensor = self.compute_output(tensor, masked_tokens)

        extra = {
            # in sequence_generator, it's ok for attn to be None just certain features not enabled
            'attn': None
        }

        return tensor, extra








