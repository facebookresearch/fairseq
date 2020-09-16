# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.models import (
    register_model,
    register_model_architecture,
)

from .s_transformer import (
    SpeechTransformerModel,
    SpeechTransformerEncoder,
    speechtransformer_base
)

from .dynamic_time_pooling import (
    DynamicTimePoolingModule
)

from .transformer_simul_trans import (
    TransformerModelSimulTrans,
)

import torch

from fairseq import checkpoint_utils

from fairseq.models.fairseq_encoder import (
    EncoderOut,
)

@register_model('speechconvtransformer_simul_trans')
class SpeechTransformerModelSimulTrans(SpeechTransformerModel):
    def __init__(self, *args):
        super().__init__(*args)
        if getattr(args[0], "train_monotonic_only", False):
            for x in self.parameters():
                x.requires_grad = False

            for layer in self.decoder.layers:
                for x in layer.encoder_attn.parameters():
                    x.requires_grad = True

    @property
    def encoder_step_size(self):
        # The unit here is miliseconds
        # TODO: now it's hard coded, make it configurable from model settings later
        return 40

    @staticmethod
    def add_args(parser):
        super(SpeechTransformerModelSimulTrans, SpeechTransformerModelSimulTrans).add_args(parser)
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--train-monotonic-only",
            action="store_true",
            default=False,
            help="",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, *extra_args):
        if getattr(args, "dynamic_time_pooling", False):
            encoder = SpeechTransformerEncoderTimePooling(args, src_dict, *extra_args)
            if getattr(args, "load_pretrained_encoder_from", None):
                encoder = checkpoint_utils.load_pretrained_component_from_model(encoder, checkpoint=args.load_pretrained_encoder_from)
            #encoder.time_pooling_module = DynamicTimePoolingModule(args)
            #if getattr(args, "load_pretrained_pre_decision_from", None):
                #encoder.time_pooling_module = checkpoint_utils.load_pretrained_component_from_model(encoder.time_pooling_module, checkpoint=args.load_pretrained_pre_decision_from)

            #if getattr(args, "freeze_pre_decision", False):
            #    for x in encoder.time_pooling_module.parameters():
            #        x.requires_grad = False

        else:
            encoder = SpeechTransformerEncoder(args, src_dict, *extra_args)
            if getattr(args, "load_pretrained_encoder_from", None):
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=args.load_pretrained_encoder_from
                )

        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):

        decoder = TransformerModelSimulTrans.build_decoder(
            args, tgt_dict, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder

    def predict_from_states(self, states):
        decoder_states = self.decoder.output_layer(
            states["decoder_features"]
        )
        lprobs = self.get_normalized_probs(
            [decoder_states[:, -1:]],
            log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        token = self.decoder.dictionary.string(index)

        return token, index[0, 0].item()

    def decision_from_states(self, states):
        '''
        This funcion take states dictionary as input, and gives the agent
        a decision of whether read a token from server. Moreover, the decoder
        states are also calculated here so we can directly generate a target
        token without recompute every thing
        '''

        self.eval()

        if len(states["indices"]["src"]) == 0:
            return 0

        src_indices = states["indices"]["src"].unsqueeze(0)

        src_lengths = torch.LongTensor([src_indices.size(1)])

        tgt_indices = torch.LongTensor(
            [
                [self.decoder.dictionary.eos()]
                + states["indices"]["tgt"]
            ]
        )

        # Update encoder states if needed
        if (
            "encoder_states" not in states or
            states["encoder_states"][0].size(1) <= states["steps"]["src"]
        ):
            encoder_out_dict = self.encoder(src_indices, src_lengths)
            states["encoder_states"] = encoder_out_dict
        else:
            encoder_out_dict = states["encoder_states"]

        # online means we still need tokens to feed the model
        states["model_states"]["online"] = not (
            states["finish_read"]
            and len(states["tokens"]["src"]) == states["steps"]["src"]
        )

        states["model_states"]["steps"] = states["steps"]

        x, outputs = self.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=encoder_out_dict,
            incremental_state=states["model_states"],
            features_only=True,
        )

        states["decoder_features"] = x

        return outputs["action"]


class SpeechTransformerEncoderTimePooling(SpeechTransformerEncoder):
    def forward(self, src_tokens, src_lengths, cls_input=None, **extra_args):
        encoder_out = super().forward(src_tokens, src_lengths)
        #if getattr(self, "time_pooling_module", None):
        #    p_trigger = torch.sigmoid(self.time_pooling_module(encoder_out))
        if cls_input is not None and "boundary" in cls_input:
            p_trigger = cls_input["boundary"]
        return EncoderOut(
            encoder_out={"encoder_state": encoder_out.encoder_out, "p_trigger": p_trigger},  # T x B x C
            encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
            encoder_embedding=p_trigger,  # B x T x C
            encoder_states=None,  # List[T x B x C]
        )

    def predict_from_states(self, states):
        decoder_states = self.decoder.output_layer(
            states["decoder_features"]
        )
        lprobs = self.get_normalized_probs(
            [decoder_states[:, -1:]],
            log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        token = self.decoder.dictionary.string(index)

        return token, index[0, 0].item()

    def decision_from_states(self, states):
        '''
        This funcion take states dictionary as input, and gives the agent
        a decision of whether read a token from server. Moreover, the decoder
        states are also calculated here so we can directly generate a target
        token without recompute every thing
        '''

        self.eval()

        if len(states["indices"]["src"]) == 0:
            return 0

        src_indices = states["indices"]["src"].unsqueeze(0)

        src_lengths = torch.LongTensor([src_indices.size(1)])

        tgt_indices = torch.LongTensor(
            [
                [self.decoder.dictionary.eos()]
                + states["indices"]["tgt"]
            ]
        )

        # Update encoder states if needed
        if (
            "encoder_states" not in states or
            states["encoder_states"][0].size(1) <= states["steps"]["src"]
        ):
            encoder_out_dict = self.encoder(src_indices, src_lengths)
            states["encoder_states"] = encoder_out_dict
        else:
            encoder_out_dict = states["encoder_states"]

        # online means we still need tokens to feed the model
        states["model_states"]["online"] = not (
            states["finish_read"]
            and len(states["tokens"]["src"]) == states["steps"]["src"]
        )

        states["model_states"]["steps"] = states["steps"]

        x, outputs = self.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=encoder_out_dict,
            incremental_state=states["model_states"],
            features_only=True,
        )

        states["decoder_features"] = x

        return outputs["action"]


@register_model_architecture(
    'speechconvtransformer_simul_trans',
    'speechconvtransformer_simul_trans_base'
)
def speechtransformer_simul_trans_base(args):
    speechtransformer_base(args)
