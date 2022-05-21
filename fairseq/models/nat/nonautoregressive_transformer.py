# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import copy
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding, base_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

logger = logging.getLogger(__name__)

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


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


@register_model("nonautoregressive_transformer")
class NATransformerModel(FairseqNATModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.use_glat = getattr(args, 'use_glat', False)
        self.use_glat_with_nat_aux = getattr(args, 'use_glat_with_nat_aux', False)
        self.use_ctc_decoder = getattr(args, 'use_ctc_decoder', False)
        self.use_ctc_beam_decoder = getattr(args, 'use_ctc_beam_decoder', False)
        self.use_deep_supervision = getattr(args, 'use_deep_supervision', False)

        if self.use_ctc_beam_decoder:
            from ctcdecode import CTCBeamDecoder
            self.ctc_decoder = CTCBeamDecoder(decoder.dictionary.symbols,
                                        model_path=None,
                                        alpha=0,
                                        beta=0,
                                        cutoff_top_n=40,
                                        cutoff_prob=1.0,
                                        beam_width=args.ctc_beam_size,
                                        num_processes=20,
                                        blank_id=decoder.dictionary.blank_index,
                                        log_probs_input=True)

        if  self.use_glat_with_nat_aux and self.use_glat:
            new_args = copy.deepcopy(args)
            new_args.decoder_layers = args.glat_nat_aux_layers
            self.aux_nat_decoder = self.build_decoder(new_args, self.decoder.dictionary, self.decoder.embed_tokens)

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        NATransformerModel.add_nat_transformer_model_args(parser)

    @staticmethod
    def add_nat_transformer_model_args(parser):
        # fmt: off
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--no-decoder-input-embedding",
            action="store_true",
            help="do not embed decoder input tokens",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )
        parser.add_argument(
            "--use-glat",
            action="store_true",
            help="enables glancing sampling from glancing transformers",
        )
        parser.add_argument(
            "--use-glat-with-nat-aux",
            action="store_true",
            help="additionally uses a separate NAT model without GLAT as an auxiliariy loss, decoder is copied for that and re-initialized. This also requires --find-unused-parameters as the aux_decoder is not used during validation for producing the loss",
        )
        parser.add_argument(
            "--glat-nat-aux-factor",
            type=float,
            help="only used when using --use-glat-with-nat-aux, sets the lambda of the additional auxiliary loss",
        )
        parser.add_argument(
            "--glat-nat-aux-layers",
            type=int,
            help="only used when using --use-glat-with-nat-aux, sets the number of layers of the additional auxiliary decoder",
        )
        parser.add_argument(
            "--use-ctc-decoder",
            action='store_true',
            help="enables the ctc decoding instead of using the length prediction",
        )
        parser.add_argument(
            "--ctc-src-upsample-scale",
            type=int,
            help="how much the target length should be upsampled by",
        )
        parser.add_argument(
            "--use-ctc-beam-decoder",
            action='store_true',
            help="whether CTC Beam search should be used",
        )
        parser.add_argument(
            "--ctc-beam-size",
            type=int,
            help="beam size of the ctc decoder",
        )
        parser.add_argument(
            "--use-deep-supervision",
            action='store_true',
            help="uses deep supervision from Huang et al., 2021 where we compute the loss at each decoder layer",
        )
        # fmt: on

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def get_ctc_target_tokens(self, tgt_tokens, prev_output_tokens, log_probs):
        """
        This method provides Viterbi aligned tgt_tokens that have the same length as the decoder inputs.
        More info sec. 3.4: https://arxiv.org/pdf/2012.15833.pdf
        """
        
        #TODO: This is C++ code from https://github.com/rosinality/imputer-pytorch/tree/master/torch_imputer that is not included in this PR
        #      Up to Fairseq team to decide how they want to import and build it
        from fairseq.models.nat.torch_imputer import best_alignment
        nonpad_positions = tgt_tokens.ne(self.pad)
        seq_lens = (nonpad_positions).sum(1)
        output_masks = prev_output_tokens.ne(self.pad)
        output_length = output_masks.sum(dim=-1)
        log_probs_T = log_probs.transpose(0, 1).float()
        best_aligns = best_alignment(log_probs_T, tgt_tokens, output_length, seq_lens, self.decoder.dictionary.blank_index, zero_infinity=True)
        best_aligns_pad = torch.tensor([a + [(tgt_tokens.shape[1] - 1)*2] * (log_probs_T.size(0) - len(a)) for a in best_aligns], device=log_probs_T.device, dtype=tgt_tokens.dtype)
        oracle_pos = best_aligns_pad.div(2, rounding_mode='trunc').clip(max=tgt_tokens.shape[1] - 1) # Needed from imputer-pytorch as seen in their get_symbol method
        oracle = tgt_tokens.gather(-1, oracle_pos)
        return oracle

    @torch.no_grad()
    def glancing_sampling(self, tgt_tokens, prev_output_tokens, encoder_out):
        """
        This implements the glancing sampling from "Glancing Transformer for Non-Autoregressive Neural Machine Translation" (Qian et al., 2021)
        Paper: https://aclanthology.org/2021.acl-long.155/
        """

        def update_glat_schedule(update_num, max_update):
                train_ratio = max(0, min(1, update_num / max_update))
                schedule = 0.5 - 0.2 * train_ratio
                return schedule

        glat_current_prob = update_glat_schedule(self.update_num, self.args.max_update)

        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )

        if self.use_ctc_decoder:
            tgt_tokens = self.get_ctc_target_tokens(tgt_tokens, prev_output_tokens, F.log_softmax(word_ins_out, -1))

        pred_tokens = word_ins_out.argmax(-1)
        nonpad_positions = tgt_tokens.ne(self.pad)
        seq_lens = (nonpad_positions).sum(1)
        same_num = ((pred_tokens == tgt_tokens) & nonpad_positions).sum(1)
        keep_prob = ((seq_lens - same_num) / seq_lens * glat_current_prob).unsqueeze(-1)
        keep_word_mask = (torch.rand(prev_output_tokens.shape, device=word_ins_out.device) < keep_prob).bool()
        glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + tgt_tokens.masked_fill(
            ~keep_word_mask, 0)

        glat_info = {
            "glat_accu": (same_num.sum() / seq_lens.sum()).item(),
            "glat_context_p": glat_current_prob,
            "glat_keep": keep_prob.mean().item()
        }

        length_loss_mult = nonpad_positions.sum().item() / seq_lens.sum().item()

        return glat_prev_output_tokens, glat_info, length_loss_mult

    def nat_auxiliary_loss(self, tgt_tokens, prev_output_tokens, encoder_out):
        """
        This idea is taken from "The Volctrans GLAT System: Non-autoregressive Translation Meets WMT21" (Qian et al., 2021)
        Paper: https://arxiv.org/pdf/2109.11247.pdf
        """
        aux_word_ins_out = self.aux_nat_decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        aux_prediction_info = {
            "out": aux_word_ins_out,
            "tgt": tgt_tokens,
            "mask": tgt_tokens.ne(self.pad),
            "ls": self.args.label_smoothing,
            "nll_loss": True,
            "factor": self.args.glat_nat_aux_factor
        }
        return aux_prediction_info

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):

        ret = {}

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # Target length based on either the length prediction or CTC
        length_tgt, length_out = self.decoder.get_length_target(encoder_out, src_tokens, tgt_tokens, normalize=False)
        length_loss_factor = self.decoder.length_loss_factor

        if self.use_ctc_decoder:
            prev_output_tokens_orig = prev_output_tokens
            prev_output_tokens = self._get_initial_outputs(src_tokens, length_tgt)

        # GLAT
        if self.use_glat and tgt_tokens is not None:
            if self.use_glat_with_nat_aux:
                aux_prev_output_tokens = prev_output_tokens_orig if self.use_ctc_decoder else prev_output_tokens
                aux_prediction_info = self.nat_auxiliary_loss(tgt_tokens, aux_prev_output_tokens, encoder_out)
                ret["aux_nat"] = aux_prediction_info

            prev_output_tokens, glat_info, length_mult = self.glancing_sampling(tgt_tokens, prev_output_tokens, encoder_out)
            length_loss_factor = length_loss_factor * length_mult
            ret.update(glat_info)

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            return_deep_supervision=self.use_deep_supervision
        )

        # Either add the CTC custom loss or populate with predictions to compute loss in nat_loss.py
        if self.use_ctc_decoder:
            ctc_info = {
                "logits": word_ins_out,
                "prev_output_tokens": prev_output_tokens,
                "targets": tgt_tokens,
                "ls": self.args.label_smoothing,
                "deep_supervision": self.use_deep_supervision
            }
            ret["ctc"] = ctc_info
        else:
            length_info = {
                "out": length_out,
                "tgt": length_tgt,
                "factor": length_loss_factor,
            }
            prediction_info = {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "deep_supervision": self.use_deep_supervision
            }
            ret["length"] = length_info
            ret["word_ins"] = prediction_info

        return ret

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        output_tokens = decoder_out.output_tokens
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        output_probs = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=decoder_out.step,
        )

        if self.use_ctc_decoder and self.use_ctc_beam_decoder:
            output_length = torch.sum(output_masks, dim=-1)
            beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(output_probs, output_length)
            top_beam_tokens = beam_results[:, 0, :]
            top_beam_len = out_lens[:, 0]
            mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len).repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
            top_beam_tokens[~mask] = self.decoder.pad
            output_tokens = top_beam_tokens.to(output_probs.device)
            output_scores = torch.full(top_beam_tokens.size(), 1.0)
        else:
            _scores, _tokens = output_probs.max(-1)
            output_scores = decoder_out.output_scores
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

    def _get_initial_outputs(self, src_tokens, length_tgt):
        """
        Returns a tensor of shape B x T where T is the maximum length target for the batch.
        It will contain elements that have the structure:
        self.bos, multiple self.unk up to target length, self.eos, multiple self.pad if needed
        """
        max_length = length_tgt.clamp_(min=2, max=self.args.max_target_positions).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), max_length).fill_(self.pad)
        initial_output_tokens.masked_fill_(idx_length[None, :] < length_tgt[:, None], self.unk)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return initial_output_tokens

    def initialize_output_tokens(self, encoder_out, src_tokens):
        """
        Initializes the DecoderOut scores and tokens with zeros and sentences containing self.unk.
        This is used to simulate previous output tokens.
        Currently this method only finds usage in the iterative_refinement_generator.py
        """
        length_tgt, _ = self.decoder.get_length_target(encoder_out, src_tokens, normalize=True)
        initial_output_tokens = self._get_initial_outputs(src_tokens, length_tgt)
        initial_output_scores = initial_output_tokens.new_zeros(*initial_output_tokens.size()).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (length_tgt[:, None] + utils.new_arange(length_tgt, 1, beam_size) - beam_size // 2)
        initial_output_tokens = self._get_initial_outputs(length_tgt, length_tgt)
        initial_output_scores = initial_output_tokens.new_zeros(*initial_output_tokens.size()).type_as(decoder_out.output_scores)
        return decoder_out._replace(output_tokens=initial_output_tokens, output_scores=initial_output_scores)


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.no_decoder_input_embedding = getattr(args, "no_decoder_input_embedding", False)
        self.use_ctc_decoder = getattr(args, 'use_ctc_decoder', False)

        if not self.use_ctc_decoder:
            self.embed_length = Embedding(getattr(args, "max_target_positions", 256), self.encoder_embed_dim, None)

        if self.no_decoder_input_embedding:
            delattr(self, "embed_tokens")

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, return_deep_supervision=False, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        if return_deep_supervision:
            deep_predictions = self.compute_deep_predictions(extra['inner_states'][1:], normalize) # skip the first element of extra since thats embedding
        else:
            decoder_out = self.output_layer(features)
            if normalize:
                decoder_out = F.log_softmax(decoder_out, -1)

        return deep_predictions if return_deep_supervision else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def get_length_target(self, encoder_out, src_tokens, tgt_tokens=None, normalize=False):
        if self.use_ctc_decoder:
            length_out = None
            length_tgt = torch.sum(src_tokens.ne(self.pad), -1) * self.args.ctc_src_upsample_scale
        else:
            length_out = self.forward_length(normalize=normalize, encoder_out=encoder_out)
            length_tgt = self.forward_length_prediction(length_out, encoder_out, tgt_tokens)
        return length_tgt, length_out

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
            src_embd = encoder_out["encoder_embedding"][0]
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None
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

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
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

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            if not self.no_decoder_input_embedding:
                x = self.embed_scale * self.embed_tokens(prev_output_tokens)
                if self.project_in_dim is not None:
                    x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            if self.no_decoder_input_embedding and states is None:
                x = positions
            else:
                x += positions

        x = self.dropout_module(x)
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

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

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

        return length_tgt

    def compute_deep_predictions(self, x, normalize):
        if x:
            if self.layer_norm:
                x[-1] = self.layer_norm(x[-1])
            x = torch.stack(x) # [decoder_layers, T, B, C]
            x = x.transpose(1,2)
            if self.project_out_dim is not None:
                x = self.project_out_dim(x)
            x = self.output_layer(x)
            if normalize:
                x = F.log_softmax(x, -1)
            return x

@register_model_architecture("nonautoregressive_transformer", "nonautoregressive_transformer")
def base_nonautoregressive_architecture(args):
    base_architecture(args)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.no_decoder_input_embedding = getattr(args, "no_decoder_input_embedding", False)

    # --- GLAT ---
    args.use_glat = getattr(args, "use_glat", False)
    args.use_glat_with_nat_aux = getattr(args, "use_glat_with_nat_aux", False)
    args.glat_nat_aux_factor = getattr(args, "glat_nat_aux_factor", 1)
    args.glat_nat_aux_layers = getattr(args, "glat_nat_aux_layers", args.decoder_layers)

    # --- CTC ---
    args.use_ctc_decoder = getattr(args, "use_ctc_decoder", False)
    args.ctc_src_upsample_scale = getattr(args, "ctc_src_upsample_scale", 2)
    args.use_ctc_beam_decoder = getattr(args, "use_ctc_beam_decoder", False)
    args.ctc_beam_size = getattr(args, "ctc_beam_size", 1)

    # --- DSLP ---
    args.use_deep_supervision = getattr(args, "use_deep_supervision", False)
