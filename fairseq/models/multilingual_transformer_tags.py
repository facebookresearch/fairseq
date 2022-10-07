# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from fairseq.data import data_utils
import torch

from typing import Optional, Tuple
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.multilingual_transformer import MultilingualTransformerModel
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
)

@register_model("multilingual_transformer_with_tags")
class TaggedMultilingualTransformerModel(MultilingualTransformerModel):
    """
    Adds tags embeddings to textual embeddings in the input.
    """
    @classmethod
    def _get_module_class(cls, is_encoder, args, lang_dict, embed_tokens, langs):
        module_class = TaggedTransformerEncoder if is_encoder else TransformerDecoder
        return module_class(args, lang_dict, embed_tokens)


TAGS = {'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART'}


class TaggedTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        super().__init__(args, dictionary, embed_tokens, return_fc)
        self.initial_tags_idxs = []
        self.end_tags_idxs = []
        for tag in TAGS:
            init_tag_idx = dictionary.index("<{}>".format(tag))
            end_tag_idx = dictionary.index("</{}>".format(tag))
            assert init_tag_idx != dictionary.unk_index, "<{}> was not found in the tgt dict".format(tag)
            assert end_tag_idx != dictionary.unk_index, "</{}> was not found in the tgt dict".format(tag)
            self.initial_tags_idxs.append(init_tag_idx)
            self.end_tags_idxs.append(end_tag_idx)

    def strip_tags_from_text(self, tokens) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Removes from the text the tags and returns:
          - a Tensor with the tokens representing the clean text
          - a Tensor indicating the ID of the tag each token belongs to (or 0 if it belongs to no tag)
        """
        clean_tokens = []
        tags = []
        current_tag_idx = None
        for idx_t in tokens:
            idx = idx_t.item()
            if idx in self.initial_tags_idxs:
                current_tag_idx = self.initial_tags_idxs.index(idx)
            elif current_tag_idx is not None and self.end_tags_idxs[current_tag_idx] == idx:
                current_tag_idx = None
            else:
                clean_tokens.append(idx)
                tags.append(self.initial_tags_idxs[current_tag_idx] if current_tag_idx is not None else self.padding_idx)
        return torch.tensor(clean_tokens).to(tokens), torch.tensor(tags).to(tokens)

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None):
        src_tokens_txt, src_tokens_tags = [], []
        for batch_idx in range(src_tokens.shape[0]):
            src_tokens_txt_item, src_tokens_tags_item = self.strip_tags_from_text(
                src_tokens[batch_idx][-src_lengths[batch_idx]:])
            src_tokens_txt.append(src_tokens_txt_item)
            src_tokens_tags.append(src_tokens_tags_item)
        src_tokens_txt = data_utils.collate_tokens(
            src_tokens_txt,
            self.padding_idx,
            left_pad=True,
        )
        src_tokens_txt.tags = data_utils.collate_tokens(
            src_tokens_tags,
            self.padding_idx,
            left_pad=True,
        )
        return super().forward(src_tokens_txt, src_lengths, return_all_hiddens, token_embeddings)

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens) + self.embed_tokens(src_tokens.tags)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed


@register_model_architecture("multilingual_transformer_with_tags", "multilingual_transformer_with_tags")
def base_multilingual_architecture(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, "share_encoder_embeddings", False)
    args.share_decoder_embeddings = getattr(args, "share_decoder_embeddings", False)
    args.share_encoders = getattr(args, "share_encoders", False)
    args.share_decoders = getattr(args, "share_decoders", False)


@register_model_architecture(
    "multilingual_transformer_with_tags", "multilingual_transformer_with_tags_iwslt_de_en"
)
def multilingual_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_multilingual_architecture(args)
