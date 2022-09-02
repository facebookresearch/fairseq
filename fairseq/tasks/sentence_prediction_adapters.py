# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import contextlib
from omegaconf import open_dict, OmegaConf

from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import (
    SentencePredictionTask,
    SentencePredictionConfig,
)


logger = logging.getLogger(__name__)


@register_task("sentence_prediction_adapters", dataclass=SentencePredictionConfig)
class SentencePredictionAdapterTask(SentencePredictionTask):
    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        model.register_classification_head(
            self.cfg.classification_head_name,
            num_classes=self.cfg.num_classes,
        )

        logger.info("Freezing Embedding Parameters")
        for parameter in model.encoder.sentence_encoder.embed_positions.parameters():
            parameter.requires_grad = False
        for (
            parameter
        ) in model.encoder.sentence_encoder.layernorm_embedding.parameters():
            parameter.requires_grad = False
        for parameter in model.encoder.sentence_encoder.embed_tokens.parameters():
            parameter.requires_grad = False

        logger.info("Freezing Adapters")
        for k, v in model.encoder.sentence_encoder.layers._modules.items():
            logger.info("Freezing Adapters in Layer " + str(k))
            if hasattr(v, "adapter_layer_norm"):
                logger.info("Freezing Adapter LN")
                for parameter in v.adapter_layer_norm.parameters():
                    parameter.requires_grad = False
            for parameter in v.adapter_modules.parameters():
                parameter.requires_grad = False

        return model
