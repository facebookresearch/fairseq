# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerModel,
)
from fairseq.modules import MultiheadAttention
from fairseq.modules.transformer_sentence_encoder import init_bert_params


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _dynamic_programming(tokens, scores):
    N, B, T = tokens.size()
    cum_scores = scores[:, :, 0].clone()  # N x B
    cum_choice = tokens.new_zeros(B, T)

    # forward
    for t in range(T - 1):
        score, choice = cum_scores.max(0)
        cum_choice[:, t] = choice
        cum_scores[0] = score + scores[0, :, t + 1]
        cum_scores[1:] = cum_scores[:-1] + scores[1:, :, t + 1]

    # back-tracking
    end_score, end_choice = cum_scores.max(0)
    cum_choice[:, T - 1] = end_choice
    for t in range(T - 2, -1, -1):
        is_start = (cum_choice[:, t + 1] == 0).type_as(cum_choice)
        cum_choice[:, t] = (cum_choice[:, t + 1] - 1) * ~is_start + cum_choice[
            :, t
        ] * is_start

    # finalize the prediction
    tokens = tokens.gather(0, cum_choice.unsqueeze(0)).squeeze(0)
    scores = scores.gather(0, cum_choice.unsqueeze(0)).squeeze(0)
    return scores, tokens


def _beam_search(tokens, scores, W=None):
    N, B, T = tokens.size()

    if (W is None) or (W > N):
        W = N


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = torch.arange(max_trg_len, device=trg_lens.device).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


@register_model("nonautoregressive_transformer")
class NATransformerModel(TransformerModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )

        # length prediction
        parser.add_argument("--src-embedding-copy", action="store_true",
                            help="copy encoder word embeddings as the initial input of the decoder")
        parser.add_argument("--pred-length-offset", action="store_true",
                            help="predicting the length difference between the target and source sentences")
        parser.add_argument("--sg-length-pred", action="store_true",
                            help="stop the gradients back-propagated from the length predictor")
        parser.add_argument("--length-loss-factor", type=float,
                            help="weights on the length prediction loss")

        # n-gram predictor
        parser.add_argument(
            "--ngram-predictor",
            nargs="?",
            const=4,
            default=1,
            type=int,
            help="adding an additional n-gram predictor.",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        length_out, length_tgt = self.decoder.forward_length_prediction(
            encoder_out, tgt_tokens
        )

        word_ins_out, word_ins_tgt, word_ins_mask = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, tgt_tokens=tgt_tokens
        )

        return {
            "word_ins_out": word_ins_out,
            "word_ins_tgt": word_ins_tgt,
            "word_ins_mask": word_ins_mask,
            "length_out": length_out,
            "length_tgt": length_tgt,
            "length_w": self.decoder.length_loss_factor,
        }

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out["step"]
        output_tokens = decoder_out["output_tokens"]
        output_scores = decoder_out["output_scores"]

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            output_tokens,
            encoder_out=encoder_out,
            decoding_format=decoding_format,
            step=step,
        )
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        return {"output_tokens": output_tokens, "output_scores": output_scores}

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        _, length_tgt = self.decoder.forward_length_prediction(encoder_out)
        max_length = length_tgt.max()
        idx_length = torch.arange(max_length, device=src_tokens.device)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"])

        return {
            "output_tokens": initial_output_tokens,
            "output_scores": initial_output_scores,
        }


class NATransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )

        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(256, self.encoder_embed_dim, None)

        self.ngram_predictor = getattr(args, "ngram_predictor", 1)
        self.ngram_layer = (
            None if (self.ngram_predictor == 1) else NgramDecoderLayer(args, True)
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        tgt_tokens=None,
        decoding_format=None,
        step=0,
        **kwargs
    ):

        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )

        if tgt_tokens is not None:
            if self.ngram_layer is None:
                word_ins_mask = tgt_tokens.ne(self.padding_idx)
                word_ins_tgt = tgt_tokens
            else:
                context_embeds, context_masks = self.forward_ngram_context(tgt_tokens)
                features = self.ngram_layer(features, context_embeds=context_embeds)
                word_ins_tgt = tgt_tokens[:, :, None].repeat(1, 1, self.ngram_predictor)
                word_ins_mask = word_ins_tgt.ne(self.padding_idx) & context_masks

            return self.output_layer(features), word_ins_tgt, word_ins_mask

        else:
            if self.ngram_layer is None:
                return F.log_softmax(self.output_layer(features), -1).max(-1)
            else:
                # inner iterations
                return self.forward_ngram_decoding(
                    features, prev_output_tokens.eq(self.padding_idx), decoding_format
                )

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy:
            src_embd = encoder_out["encoder_embedding"]
            src_mask = encoder_out["encoder_padding_mask"]
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
            )

            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens,
                self.forward_copying_source(
                    src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                ),
            )

        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn = layer(
                x,
                encoder_out["encoder_out"] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"]
                if encoder_out is not None
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_ngram_context(self, tgt_tokens):
        tgt_embeds = self.forward_embedding(tgt_tokens)
        n_contexts = self.ngram_predictor - 1

        # shifting the embeddings
        # context_embeds: N x B x T x C
        # context_masks:  B x T x N
        context_embeds = tgt_embeds.new_zeros(n_contexts, *tgt_embeds.size())
        context_masks = tgt_embeds.new_ones(
            *tgt_embeds.size()[:2], self.ngram_predictor
        ).bool()

        for k in range(n_contexts):
            context_embeds[k, :, k + 1:] = tgt_embeds[:, : -k - 1]
            context_masks[:, : k + 1, k + 1] = 0

        return context_embeds, context_masks

    def forward_ngram_decoding(self, features, padding_mask=None, decoding_format=None):
        context_embeds = None
        scores, tokens = [], []
        ensemble_score = None
        ensemble_index = None

        if decoding_format is None:
            decoding_format = "ensemble"

        for k in range(self.ngram_predictor):
            ngram_out = self.ngram_layer(
                features, context_embeds=context_embeds, incremental=True
            )
            ngram_scores = F.log_softmax(self.output_layer(ngram_out), -1)
            max_score, max_token = ngram_scores.max(-1)

            if decoding_format == "vote":
                ngram_scores = _argmax(ngram_scores, -1)

            if ensemble_score is None:
                ensemble_score = ngram_scores
                ensemble_index = ensemble_score.new_ones(*ensemble_score.size()[:2])
            else:
                ensemble_index[:, k:] = ensemble_index[:, k:] + 1
                ensemble_score = ensemble_score + ngram_scores.masked_fill_(
                    (ensemble_index < k)
                    .unsqueeze(2)
                    .repeat(1, 1, ensemble_score.size(2)),
                    0,
                )
                max_score[:, :k] = float("-inf")

            if decoding_format == "unigram":
                break

            scores.append(max_score.masked_fill_(padding_mask, 0))
            tokens.append(max_token.masked_fill_(padding_mask, self.padding_idx))

            # context_embeds: N x B x T x C
            if context_embeds is None:
                context_embeds = self.forward_embedding(max_token).unsqueeze(0)

            else:
                context_embeds = torch.cat(
                    [self.forward_embedding(max_token).unsqueeze(0), context_embeds], 0
                )

            context_embeds[:, :, 1:] = context_embeds[:, :, :-1]

        if decoding_format != "dp":
            ensemble_score = ensemble_score / ensemble_index.unsqueeze(2)
            return ensemble_score.max(-1)

        else:
            tokens = torch.cat([t.unsqueeze(0) for t in tokens], 0)
            scores = torch.cat([s.unsqueeze(0) for s in scores], 0)
            return _dynamic_programming(tokens, scores)

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_length_prediction(self, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"]  # T x B x C
        src_masks = encoder_out["encoder_padding_mask"]  # B x T or None

        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()

        length_out = F.linear(enc_feats, self.embed_length.weight)

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_out, length_tgt


class NgramDecoderLayer(TransformerDecoderLayer):
    """
    N-gram Decoder Layer:

    This module can be pluged in the last layer of any Non-autoregressive Model's
    It provides an alternative way to capture local n-gram information by running the block multiple times.
    """

    def __init__(self, args, no_encoder_attn=False):
        super(NgramDecoderLayer, self).__init__(args, no_encoder_attn=no_encoder_attn)
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=1,  # maybe n-gram does not need too many heads.
            dropout=args.attention_dropout,
            self_attention=False,
            encoder_decoder_attention=True,
        )

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        context_embeds=None,
        incremental=False,
    ):
        # x: T x B x C
        # context_embeds: N x T x B x C
        T, B, C = x.size()

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x = x.contiguous().view(1, T * B, C).contiguous()

        if context_embeds is not None:
            N = context_embeds.size(0)
            context_embeds = context_embeds.view(N, T * B, C).contiguous()

        if not incremental:
            assert context_embeds is not None, "we need context for training"
            # attn_weights: (n_head x T x B) x 1 x N
            # v: (n_head x T x B) x N x (dim / n_head)
            # -- move the attention computation outside --
            attn_weights, values = self.self_attn(
                query=x, key=context_embeds, value=context_embeds, before_softmax=True
            )

            attn_weights = attn_weights.repeat(1, N, 1)
            attn_masks = attn_weights.new_ones(N, N).triu_(1).bool()
            attn_masks = attn_masks.unsqueeze(0).repeat(attn_weights.size(0), 1, 1)

            attn_weights = attn_weights.masked_fill(attn_masks, float("-inf"))
            attn_weights = utils.softmax(attn_weights, dim=-1).type_as(attn_weights)
            attn_weights = F.dropout(
                attn_weights, p=self.self_attn.dropout, training=self.training
            )

            # (n_head x T x B) x N x (dim / n_head)
            attn = torch.bmm(attn_weights, values)
            attn = attn.transpose(0, 1).contiguous()
            attn = attn.view(N, T * B, C).contiguous()
            attn = attn.transpose(1, 0).contiguous()
            attn = attn.view(T, B, N, C)

            residual = residual.unsqueeze(2)
            x = self.self_attn.out_proj(attn)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.cat([residual, residual + x], 2)

        else:
            if context_embeds is None:
                x = residual

            else:
                x, _ = self.self_attn(query=x, key=context_embeds, value=context_embeds)
                x = x.view(T, B, C)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = residual + x

        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            raise NotImplementedError

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x


@register_model_architecture(
    "nonautoregressive_transformer", "nonautoregressive_transformer"
)
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

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)


@register_model_architecture(
    "nonautoregressive_transformer", "nonautoregressive_transformer_wmt_en_de"
)
def nonautoregressive_transformer_wmt_en_de(args):
    base_architecture(args)
