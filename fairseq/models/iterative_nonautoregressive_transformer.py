# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nonautoregressive_transformer import NATransformerModel


def _sequential_poisoning(s, V, beta=0.33, bos=2, eos=3, pad=1):
    # s: input batch
    # V: vocabulary size
    rand_words = torch.randint(low=4, high=V, size=s.size(), device=s.device)
    choices = torch.rand(size=s.size(), device=s.device)
    choices.masked_fill_((s == pad) | (s == bos) | (s == eos), 1)

    replace = choices < beta / 3
    repeat = (choices >= beta / 3) & (choices < beta * 2 / 3)
    swap = (choices >= beta * 2 / 3) & (choices < beta)
    safe = choices >= beta

    for i in range(s.size(1) - 1):
        rand_word = rand_words[:, i]
        next_word = s[:, i + 1]
        self_word = s[:, i]

        replace_i = replace[:, i]
        swap_i = swap[:, i] & (next_word != 3)
        repeat_i = repeat[:, i] & (next_word != 3)
        safe_i = safe[:, i] | ((next_word == 3) & (~replace_i))

        s[:, i] = (
            self_word * (safe_i | repeat_i).long()
            + next_word * swap_i.long()
            + rand_word * replace_i.long()
        )
        s[:, i + 1] = (
            next_word * (safe_i | replace_i).long()
            + self_word * (swap_i | repeat_i).long()
        )
    return s


def gumbel_noise(input, TINY=1e-8):
    return input.new_zeros(*input.size()).uniform_().add_(
        TINY).log_().neg_().add_(TINY).log_().neg_()


@register_model("iterative_nonautoregressive_transformer")
class IterNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument("--train-step", type=int,
                            help="number of refinement iterations during training")
        parser.add_argument("--dae-ratio", type=float,
                            help="the probability of switching to the denoising auto-encoder loss")
        parser.add_argument("--stochastic-approx", action="store_true",
                            help="sampling from the decoder as the inputs for next iteration")

    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        model.train_step = getattr(args, "train_step", 4)
        model.dae_ratio = getattr(args, "dae_ratio", 0.5)
        model.stochastic_approx = getattr(args, "stochastic_approx", False)
        return model

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):

        B, T = prev_output_tokens.size()

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        length_out, length_tgt = self.decoder.forward_length_prediction(
            encoder_out, tgt_tokens
        )
        word_ins_outs, word_ins_tgts, word_ins_masks = [], [], []
        for t in range(self.train_step):
            word_ins_out, word_ins_tgt, word_ins_mask = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                tgt_tokens=tgt_tokens,
                step=t,
            )

            word_ins_outs.append(word_ins_out)
            word_ins_tgts.append(word_ins_tgt)
            word_ins_masks.append(word_ins_mask)

            if t < (self.train_step - 1):
                # prediction for next iteration
                if self.stochastic_approx:
                    word_ins_prediction = (
                        word_ins_out + gumbel_noise(word_ins_out)
                    ).max(-1)[1]
                else:
                    word_ins_prediction = word_ins_out.max(-1)[1]

                prev_output_tokens = prev_output_tokens.masked_scatter(
                    word_ins_mask, word_ins_prediction[word_ins_mask]
                )

                if self.dae_ratio > 0:
                    # we do not perform denoising for the first iteration
                    corrputed = (
                        torch.rand(size=(B,), device=prev_output_tokens.device)
                        < self.dae_ratio
                    )
                    corrputed_tokens = _sequential_poisoning(
                        tgt_tokens[corrputed],
                        len(self.tgt_dict),
                        0.33,
                        self.bos,
                        self.eos,
                        self.pad,
                    )
                    prev_output_tokens[corrputed] = corrputed_tokens

        # concat everything
        word_ins_out = torch.cat(word_ins_outs, 0)
        word_ins_tgt = torch.cat(word_ins_tgts, 0)
        word_ins_mask = torch.cat(word_ins_masks, 0)

        return {
            "word_ins": {
                "out": word_ins_out, "tgt": word_ins_tgt,
                "mask": word_ins_mask, "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }
        }


@register_model_architecture(
    "iterative_nonautoregressive_transformer", "iterative_nonautoregressive_transformer"
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
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

    args.train_step = getattr(args, "train_step", 4)
    args.dae_ratio = getattr(args, "dae_ratio", 0.5)
    args.stochastic_approx = getattr(args, "stochastic_approx", False)


@register_model_architecture(
    "iterative_nonautoregressive_transformer",
    "iterative_nonautoregressive_transformer_wmt_en_de",
)
def iter_nat_wmt_en_de(args):
    base_architecture(args)
