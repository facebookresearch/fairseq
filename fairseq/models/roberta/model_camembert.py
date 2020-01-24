# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
CamemBERT: a Tasty French Language Model
"""

from fairseq.models import register_model

from .hub_interface import RobertaHubInterface
from .model import RobertaModel


@register_model('camembert')
class CamembertModel(RobertaModel):

    @classmethod
    def hub_models(cls):
        return {
            'camembert.v0': 'http://dl.fbaipublicfiles.com/fairseq/models/camembert.v0.tar.gz',
        }

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='sentencepiece', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])
