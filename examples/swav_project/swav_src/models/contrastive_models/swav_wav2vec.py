
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from fairseq.models import register_model
from fairseq.modules import (
    GradMultiply,
)
from fairseq.utils import is_xla_tensor
from omegaconf import MISSING

from fairseq.models.wav2vec import Wav2Vec2Config, Wav2Vec2Model

import logging

logger = logging.getLogger(__name__)


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


def upgrade_state_dict_named_prototypes(model, state_dict, name):
    if "prototypes.weight" not in state_dict:
        prot_state = model.prototypes.state_dict()
        for k, v in prot_state.items():
            state_dict[f'prototypes.{k}'] = v
        # logger.warning('Overwriting checkpoint prototypes.weight with current state')


def universal_build_prototypes(cfg):
    nmb_prototypes = getattr(cfg, 'nmb_prototypes', "-1")
    nmb_prototypes = [int(x) for x in nmb_prototypes.split(",")]
    if isinstance(nmb_prototypes, list):
        if len(nmb_prototypes) > 1:
            prototypes = MultiPrototypes(cfg.encoder_embed_dim, nmb_prototypes)
        else:
            prototypes = nn.Linear(cfg.encoder_embed_dim, nmb_prototypes[0], bias=False)
    elif nmb_prototypes > 0:
        prototypes = nn.Linear(cfg.encoder_embed_dim, nmb_prototypes, bias=False)
    return prototypes


@dataclass
class SwavWav2Vec2Config(Wav2Vec2Config):
    nmb_prototypes: str = field(
        default="3000",
        metadata={"help": "number prototypes"},
    )
    prot_hidden: str = field(
        default="avg_pooling",
        metadata={"help": "prototype hidden"},
    )
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    # this holds the loaded wav2vec args
    w2v_args: Any = None


@register_model("swav_wav2vec2", dataclass=SwavWav2Vec2Config)
class SwavWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, cfg: SwavWav2Vec2Config):
        super().__init__(cfg)
        self.prototypes = universal_build_prototypes(cfg)
        self.num_updates = None

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @staticmethod
    def add_args(parser):
        Wav2Vec2Model.add_args(parser=parser)
        # add_swav_args(parser)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        upgrade_state_dict_named_prototypes(self, state_dict, name)
    
    def extract_prot_hidden(self, encoder_output, padding_mask=None, **kwargs):
        """Encoder output: B x T x C"""
        prot_hidden = getattr(self.cfg, 'prot_hidden', 'bos')
        if prot_hidden == "bos":
            raise ValueError(f'we prefer avg_pooling for mBART, got {prot_hidden}')
            return encoder_output[0]  # take <s> token (equiv. to [CLS])
            # 
        elif prot_hidden == "eos":
            # eos: int = self.eos
            raise ValueError(f'we prefer avg_pooling for mBART, got {prot_hidden}')

            # return sentence_representation
        elif prot_hidden == "global_pooling":
            raise NotImplementedError(f'prot_hidden: {self.cfg.prot_hidden} not impl')
        
        elif prot_hidden == "avg_pooling":
            _x_prot_feed = encoder_output
            if padding_mask is not None:
                input_lengths = (1 - padding_mask.long()).sum(-1)
                # encoder_output already B x T x C
                mask = torch.arange(_x_prot_feed.size(1)).to(_x_prot_feed).unsqueeze_(0) < (input_lengths - 1).unsqueeze_(-1)
                mask = mask.unsqueeze_(-1)
                _x_prot_feed = _x_prot_feed * mask
                sent_embed = (_x_prot_feed / mask.sum(1, keepdim=True)).sum(1)
            else:
                sent_embed = (_x_prot_feed / _x_prot_feed.size(1)).sum(1)
            return sent_embed
        else:
            raise ValueError(f'prot_hidden: {self.cfg.prot_hidden} not found')
    
    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
        get_prototypes: bool = False, 
        get_prototypes_only: bool = False,
        pre_norm_prototypes: bool = False,
        **kwargs,
    ):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        
        if pre_norm_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)

        prot_extra = {}
        if get_prototypes or get_prototypes_only:
            # x: B x T x C
            x_feature = x
            _embed = self.extract_prot_hidden(x_feature, padding_mask)
            prot_embed = _embed.detach()
            prot_out = self.prototypes(_embed)
            assert prot_out.dim() == 2, f'prot_out dim not 2 {prot_out.size()}, {_embed.size()}'
            assert not torch.any(torch.isinf(prot_out))
            prot_extra['prot_out'] = prot_out
            prot_extra['prot_embed'] = prot_embed
            if get_prototypes_only:
                # NOTE: WARNING this will cause decoder params to be unused!
                return {
                    'prot_out': prot_out, 
                    'prot_embed': prot_embed
                }
        else:
            # add dummy prototypes to prevent error
            try:
                x += self.prototypes(x.new(1, x.size(-1)).fill_(0))[0, 0] * 0
            except Exception as e:
                logger.warning(f'{type(x)}')
                logger.warning(f'{x.size()}')
                raise e

        if features_only:
            feature_only_out = {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }
            for k, v in prot_extra.items():
                feature_only_out[k] = v
            return feature_only_out

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands = self.quantizer(unmasked_features, produce_targets=False)[
                    "x"
                ]
                negs, _ = self.sample_negatives(
                    neg_cands,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        if not is_xla_tensor(x):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp
        
        for k, v in prot_extra.items():
            result[k] = v

        return result

