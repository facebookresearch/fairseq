# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F

from fairseq.models import register_model, register_model_architecture
from fairseq.models.levenshtein_transformer import (
    LevenshteinTransformerDecoder,
    LevenshteinTransformerModel,
)
from fairseq.models.transformer import Linear, TransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params


class NegativeDistanceScore(object):
    def __init__(self):

        # pre-compute some values
        self.scores = {}

        self.scores[0.5] = self.compute_score_full(50, 0.5)
        self.scores[1.0] = self.compute_score_full(50, 1.0)
        self.scores[2.0] = self.compute_score_full(50, 2.0)

    def __call__(self, i, L, tau):
        if (tau is None) or (tau > 1000):
            return 1 / L

        if tau in self.scores:
            if L < self.scores[tau].shape[0]:
                return self.scores[tau][L - 1, i]
        return self.compute_score(L, tau)[i]

    def compute_score(self, L, tau):
        s = np.array([-abs(L / 2 - i) / tau for i in range(L)])
        s = np.exp(s - s.max())
        return s / s.sum()

    def compute_score_full(self, L, tau):
        s = -abs(np.arange(0, L - 1)[:, None] / 2 - np.arange(L)[None, :]) / tau
        s = np.tril(s, 0) + np.triu(s - float("inf"), 1)
        s = np.exp(s - s.max(1, keepdims=True))
        return s / s.sum(1, keepdims=True)


neg_scorer = NegativeDistanceScore()


def _get_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx, vocab_size, tau=None):
    try:
        from fairseq import libnat
    except ImportError as e:
        import sys
        sys.stderr.write('ERROR: missing libnat. run `pip install --editable .`\n')
        raise e

    B = in_tokens.size(0)
    T = in_tokens.size(1)
    V = vocab_size

    with torch.cuda.device_of(in_tokens):
        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )
    insert_labels = [a[:-1] for a in full_labels]

    # numericalize1
    insert_label_tensors = in_tokens.new_zeros(B * (T - 1) * V).float()
    insert_index, insert_labels = zip(
        *[
            (w + (j + i * (T - 1)) * V, neg_scorer(k, len(label), tau))
            for i, labels in enumerate(insert_labels)
            for j, label in enumerate(labels[1:-1])
            for k, w in enumerate(label)
        ]
    )  # HACK 1:-1
    insert_index, insert_labels = [
        torch.tensor(list(a), device=in_tokens.device)
        for a in [insert_index, insert_labels]
    ]
    insert_label_tensors.scatter_(0, insert_index.long(), insert_labels)
    insert_label_tensors = insert_label_tensors.view(B, T - 1, V)

    return insert_label_tensors


def _apply_ins_words(in_tokens, in_scores, word_ins_pred, word_ins_scores, padding_idx):

    padding_masks = in_tokens[:, 1:].eq(padding_idx)
    word_ins_scores.masked_fill_(padding_masks, 0.0)
    word_ins_pred.masked_fill_(padding_masks, padding_idx)

    in_coords = torch.arange(in_tokens.size(1), device=in_tokens.device)
    in_coords = in_coords.unsqueeze(0).repeat(in_tokens.size(0), 1).type_as(in_scores)

    # shift all padding predictions to infinite
    out_coords = (in_coords[:, 1:] - 0.5).masked_fill(
        word_ins_pred.eq(padding_idx), float("inf")
    )
    out_coords = torch.cat([in_coords, out_coords], 1).sort(-1)[1]
    out_tokens = torch.cat([in_tokens, word_ins_pred], 1).gather(1, out_coords)
    out_scores = torch.cat([in_scores, word_ins_scores], 1).gather(1, out_coords)
    return out_tokens, out_scores


@register_model("insertion_transformer")
class InsertionTransformerModel(LevenshteinTransformerModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )
        parser.add_argument("--label-tau", default=None, type=float)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = InsertionTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # generate training labels for insertion
        word_ins_out = self.decoder.forward_word_ins(
            prev_output_tokens, encoder_out=encoder_out
        )
        word_ins_tgt = _get_ins_targets(
            prev_output_tokens,
            tgt_tokens,
            self.pad,
            self.unk,
            len(self.tgt_dict),
            tau=self.decoder.label_tau,
        ).type_as(word_ins_out)
        word_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

        return {
            "word_ins_out": word_ins_out,
            "word_ins_tgt": word_ins_tgt,
            "word_ins_mask": word_ins_masks,
        }

    def forward_decoder(
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out["output_tokens"]
        output_scores = decoder_out["output_scores"]
        # TODO: decoding for InsertionTransformer
        word_ins_out = self.decoder.forward_word_ins(
            output_tokens, encoder_out=encoder_out
        )
        word_ins_score = F.log_softmax(word_ins_out, 2)
        if eos_penalty > 0.0:
            word_ins_score[:, :, self.pad] -= eos_penalty
        word_ins_score, word_ins_pred = word_ins_score.max(-1)
        output_tokens, output_scores = _apply_ins_words(
            output_tokens, output_scores, word_ins_pred, word_ins_score, self.pad
        )

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        return {"output_tokens": output_tokens, "output_scores": output_scores}


class InsertionTransformerDecoder(LevenshteinTransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        # use the TransformerDecoder's __init__
        super(LevenshteinTransformerDecoder, self).__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )

        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.pool_out = Linear(self.output_embed_dim * 2, self.output_embed_dim)

        self.label_tau = getattr(args, "label_tau", None)

    def forward_word_ins(self, prev_output_tokens, encoder_out=None):
        features, _ = self.extract_features(prev_output_tokens, encoder_out=encoder_out)
        features = self.pool_out(
            torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        )
        return self.output_layer(features)

    def forward_mask_ins(self, *args, **kwargs):
        raise NotImplementedError

    def forward_word_del(self, *args, **kwargs):
        raise NotImplementedError

    def forward_word_del_mask_ins(self, *args, **kwargs):
        raise NotImplementedError


@register_model_architecture("insertion_transformer", "insertion_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # special for insertion transformer
    args.label_tau = getattr(args, "label_tau", None)
