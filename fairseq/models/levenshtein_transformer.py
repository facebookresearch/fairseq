# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    TransformerDecoderLayer
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import new_arange

# -------------- Helper Functions --------------------------------------------------- #
def _skip(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [_skip(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: _skip(v, mask) for k, v in x.items()}

    raise NotImplementedError


def _skip_encoder_out(encoder, encoder_out, mask):
    if not mask.any():
        return encoder_out
    else:
        return encoder.reorder_encoder_out(encoder_out, mask.nonzero().squeeze())


def _fill(x, mask, y, padding_idx):
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None:
        return y
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))
    n_selected = mask.sum()
    assert n_selected == y.size(0)

    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        dims = [x.size(0), y.size(1) - x.size(1)]
        if x.dim() == 3:
            dims.append(x.size(2))
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 1)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = padding_idx
        if x.dim() == 2:
            x[mask, :y.size(1)] = y
        else:
            x[mask, :y.size(1), :] = y
    else:
        x[mask] = y
    return x


def load_libnat():
    try:
        from fairseq import libnat
    except ImportError as e:
        import sys
        sys.stderr.write("ERROR: missing libnat. run `pip install --editable .`\n")
        raise e
    return libnat


def _get_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx):
    libnat = load_libnat()

    in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

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
    mask_inputs = [
        [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
    ]

    # generate labels
    masked_tgt_masks = []
    for mask_input in mask_inputs:
        mask_label = []
        for beam_size in mask_input[1:-1]:  # HACK 1:-1
            mask_label += [0] + [1 for _ in range(beam_size)]
        masked_tgt_masks.append(
            mask_label + [0 for _ in range(out_seq_len - len(mask_label))]
        )
    mask_ins_targets = [
        mask_input[1:-1] + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
        for mask_input in mask_inputs
    ]

    # transform to tensor
    masked_tgt_masks = torch.tensor(
        masked_tgt_masks, device=out_tokens.device
    ).bool()
    mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
    masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
    return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets


def _get_del_targets(in_tokens, out_tokens, padding_idx):
    libnat = load_libnat()

    out_seq_len = out_tokens.size(1)

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
    word_del_targets = [b[-1] for b in full_labels]
    word_del_targets = [
        labels + [0 for _ in range(out_seq_len - len(labels))]
        for labels in word_del_targets
    ]

    # transform to tensor
    word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)
    return word_del_targets


def _get_del_ins_targets(in_tokens, out_tokens, padding_idx):
    libnat = load_libnat()

    in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

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

    word_del_targets = [b[-1] for b in full_labels]
    word_del_targets = [
        labels + [0 for _ in range(out_seq_len - len(labels))]
        for labels in word_del_targets
    ]

    mask_inputs = [
        [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
    ]
    mask_ins_targets = [
        mask_input[1:-1] + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
        for mask_input in mask_inputs
    ]

    # transform to tensor
    mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
    word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)
    return word_del_targets, mask_ins_targets


def _apply_ins_masks(
    in_tokens, in_scores, mask_ins_pred, padding_idx, unk_idx, eos_idx
):

    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(1)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos_idx)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)

    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = (
        new_arange(out_lengths, out_max_len)[None, :]
        < out_lengths[:, None]
    )

    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), out_max_len)
        .fill_(padding_idx)
        .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, 0] = in_scores[:, 0]
        out_scores.scatter_(1, reordering, in_scores[:, 1:])

    return out_tokens, out_scores


def _apply_ins_words(
    in_tokens, in_scores, word_ins_pred, word_ins_scores, unk_idx
):
    word_ins_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])

    if in_scores is not None:
        out_scores = in_scores.masked_scatter(
            word_ins_masks, word_ins_scores[word_ins_masks]
        )
    else:
        out_scores = None

    return out_tokens, out_scores


def _apply_del_words(
    in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx
):
    # apply deletion to a tensor
    in_masks = in_tokens.ne(padding_idx)
    bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)

    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)

    reordering = (
        new_arange(in_tokens)
        .masked_fill_(word_del_pred, max_len)
        .sort(1)[1]
    )

    out_tokens = in_tokens.masked_fill(word_del_pred, padding_idx).gather(1, reordering)

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0).gather(1, reordering)

    out_attn = None
    if in_attn is not None:
        _mask = word_del_pred[:, :, None].expand_as(in_attn)
        _reordering = reordering[:, :, None].expand_as(in_attn)
        out_attn = in_attn.masked_fill(_mask, 0.).gather(1, _reordering)

    return out_tokens, out_scores, out_attn

# ------------------------------------------------------------------------------------- #

@register_model("levenshtein_transformer")
class LevenshteinTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
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
        parser.add_argument(
            "--early-exit",
            default="6,6,6",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help="share the parameters for both mask-predictor and discriminator",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action='store_true',
            help='instead of argmax, use sampling to predict the tokens'
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LevenshteinTransformerDecoder(args, tgt_dict, embed_tokens)
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

        assert tgt_tokens is not None, "forward function only supports training."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            prev_output_tokens, tgt_tokens, self.pad, self.unk
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
        mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            prev_output_tokens, encoder_out=encoder_out
        )
        word_ins_out, _ = self.decoder.forward_word_ins(
            masked_tgt_tokens, encoder_out=encoder_out
        )

        # make online prediction
        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1).view(
                    word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        word_del_out, _ = self.decoder.forward_word_del(
            word_predictions, encoder_out)
        word_del_masks = word_predictions.ne(self.pad)

        return {
            "mask_ins_out": mask_ins_out,
            "mask_ins_tgt": mask_ins_targets,
            "mask_ins_mask": mask_ins_masks,
            "word_ins_out": word_ins_out,
            "word_ins_tgt": tgt_tokens,
            "word_ins_mask": masked_tgt_masks,
            "word_del_out": word_del_out,
            "word_del_tgt": word_del_targets,
            "word_del_mask": word_del_masks,
        }

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    def forward_decoder(
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(1)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            word_del_out, word_del_attn = self.decoder.forward_word_del(
                _skip(output_tokens, can_del_word),
                _skip_encoder_out(self.encoder, encoder_out, can_del_word)
            )
            word_del_score = F.log_softmax(word_del_out, 2)
            word_del_pred = word_del_score.max(-1)[1].bool()

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                word_del_attn,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_out, _ = self.decoder.forward_mask_ins(
                _skip(output_tokens, can_ins_mask),
                _skip_encoder_out(self.encoder, encoder_out, can_ins_mask)
            )
            mask_ins_score = F.log_softmax(mask_ins_out, 2)
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            )

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_out, word_ins_attn = self.decoder.forward_word_ins(
                _skip(output_tokens, can_ins_word),
                _skip_encoder_out(self.encoder, encoder_out, can_ins_word)
            )
            word_ins_score, word_ins_pred = F.log_softmax(word_ins_out, 2).max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens[:, 1] = self.eos

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)
        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )


class LevenshteinTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(',')]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList([
                                    TransformerDecoderLayer(args, no_encoder_attn)
                                    for _ in range(self.early_exit[1])
                                ])
        self.layers_del = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_del = nn.ModuleList([
                                    TransformerDecoderLayer(args, no_encoder_attn)
                                    for _ in range(self.early_exit[0])
                                ])

        if getattr(args, "share_discriminator_maskpredictor", False):
            assert getattr(args, "no_share_discriminator", False), "must set saperate discriminator"
            self.layers_msk = self.layers_del

    def extract_features(
        self, prev_output_tokens, encoder_out=None, early_exit=None, layers=None, **unused
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
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[: early_exit]):
            x, attn = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
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

    def forward_mask_ins(self, prev_output_tokens, encoder_out=None, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[1], layers=self.layers_msk, **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        return F.linear(features_cat, self.embed_mask_ins.weight), extra['attn']

    def forward_word_ins(self, prev_output_tokens, encoder_out=None, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[2], layers=self.layers, **unused
        )
        return self.output_layer(features), extra['attn']

    def forward_word_del(self, prev_output_tokens, encoder_out=None, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[0], layers=self.layers_del, **unused
        )
        return F.linear(features, self.embed_word_del.weight), extra['attn']


@register_model_architecture("levenshtein_transformer", "levenshtein_transformer")
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
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(args, "share_discriminator_maskpredictor", False)
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)


@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de"
)
def levenshtein_transformer_wmt_en_de(args):
    base_architecture(args)


# similar parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_vaswani_wmt_en_de_big"
)
def levenshtein_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de_big"
)
def levenshtein_transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    levenshtein_transformer_vaswani_wmt_en_de_big(args)
