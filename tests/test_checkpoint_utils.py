# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import tempfile
import unittest
from io import StringIO

from fairseq import checkpoint_utils

from tests.utils import (
    create_dummy_data,
    preprocess_translation_data,
    train_translation_model,
)


class TestCheckpointUtils(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @contextlib.contextmanager
    def _train_transformer(self, seed, extra_args=None):
        if extra_args is None:
            extra_args = []
        with tempfile.TemporaryDirectory(f"_train_transformer_seed{seed}") as data_dir:
            create_dummy_data(data_dir)
            preprocess_translation_data(data_dir)
            train_translation_model(
                data_dir,
                "transformer_iwslt_de_en",
                [
                    "--encoder-layers",
                    "3",
                    "--decoder-layers",
                    "3",
                    "--encoder-embed-dim",
                    "8",
                    "--decoder-embed-dim",
                    "8",
                    "--seed",
                    str(seed),
                ]
                + extra_args,
            )
            yield os.path.join(data_dir, "checkpoint_last.pt")

    def test_load_model_ensemble_and_task(self):
        with contextlib.redirect_stdout(StringIO()):
            with self._train_transformer(seed=123) as model1:
                with self._train_transformer(seed=456) as model2:
                    ensemble, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                        filenames=[model1, model2]
                    )
                    self.assertEqual(len(ensemble), 2)

                    # after Transformer has been migrated to Hydra, this will probably
                    # become cfg.common.seed
                    self.assertEqual(ensemble[0].args.seed, 123)
                    self.assertEqual(ensemble[1].args.seed, 456)

                    # the task from the first model should be returned
                    self.assertEqual(task.args.seed, 123)

    def test_prune_state_dict(self):
        with contextlib.redirect_stdout(StringIO()):
            extra_args = ["--encoder-layerdrop", "0.01", "--decoder-layerdrop", "0.01"]
            with self._train_transformer(seed=1, extra_args=extra_args) as model:
                ensemble, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                    filenames=[model],
                    arg_overrides={
                        "encoder_layers_to_keep": "0,2",
                        "decoder_layers_to_keep": "1",
                    },
                )
                self.assertEqual(len(ensemble), 1)
                self.assertEqual(len(ensemble[0].encoder.layers), 2)
                self.assertEqual(len(ensemble[0].decoder.layers), 1)


if __name__ == "__main__":
    unittest.main()
