# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
GottBERT: a pure German Language Model
"""

from fairseq.models import register_model

from .hub_interface import RobertaHubInterface
from .model import RobertaModel


@register_model('gottbert')
class GottbertModel(RobertaModel):

    @classmethod
    def hub_models(cls):
        return {
            'gottbert-base': 'https://dl.gottbert.de/fairseq/models/gottbert-base.tar.gz',
        }

    @classmethod
    def from_pretrained(cls,
                        model_name_or_path,
                        checkpoint_file='model.pt',
                        data_name_or_path='.',
                        bpe='hf_byte_bpe',
                        bpe_vocab='vocab.json',
                        bpe_merges='merges.txt',
                        bpe_add_prefix_space=False,
                        **kwargs
                        ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            bpe_vocab=bpe_vocab,
            bpe_merges=bpe_merges,
            bpe_add_prefix_space=bpe_add_prefix_space,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])
