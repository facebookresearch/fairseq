# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATransformerModel, base_architecture
from fairseq.modules import DynamicCRF


@register_model("nacrf_transformer")
class NACRFTransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.crf_layer = DynamicCRF(
            num_embedding=len(self.tgt_dict),
            low_rank=args.crf_lowrank_approx,
            beam_size=args.crf_beam_approx,
        )

    @property
    def allow_ensemble(self):
        return False

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument(
            "--crf-lowrank-approx",
            type=int,
            help="the dimension of low-rank approximation of transition",
        )
        parser.add_argument(
            "--crf-beam-approx",
            type=int,
            help="the beam size for apporixmating the normalizing factor",
        )
        parser.add_argument(
            "--word-ins-loss-factor",
            type=float,
            help="weights on NAT loss used to co-training with CRF loss.",
        )

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        word_ins_tgt, word_ins_mask = tgt_tokens, tgt_tokens.ne(self.pad)

        # compute the log-likelihood of CRF
        crf_nll = -self.crf_layer(word_ins_out, word_ins_tgt, word_ins_mask)
        crf_nll = (crf_nll / word_ins_mask.type_as(crf_nll).sum(-1)).mean()

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": word_ins_tgt,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": self.args.word_ins_loss_factor,
            },
            "word_crf": {"loss": crf_nll},
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder and get emission scores
        output_masks = output_tokens.ne(self.pad)
        word_ins_out = self.decoder(
            normalize=False, prev_output_tokens=output_tokens, encoder_out=encoder_out
        )

        # run viterbi decoding through CRF
        _scores, _tokens = self.crf_layer.forward_decoder(word_ins_out, output_masks)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )


@register_model_architecture("nacrf_transformer", "nacrf_transformer")
def nacrf_base_architecture(args):
    args.crf_lowrank_approx = getattr(args, "crf_lowrank_approx", 32)
    args.crf_beam_approx = getattr(args, "crf_beam_approx", 64)
    args.word_ins_loss_factor = getattr(args, "word_ins_loss_factor", 0.5)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    base_architecture(args)
