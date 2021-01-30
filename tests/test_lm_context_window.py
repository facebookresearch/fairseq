# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from fairseq.data import MonolingualDataset
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from tests import utils as test_utils


class TestLMContextWindow(unittest.TestCase):

    def test_eval_dataloader(self):
        dictionary = test_utils.dummy_dictionary(10)
        assert len(dictionary) == 14  # 4 extra special symbols
        assert dictionary.pad() == 1

        dataset = test_utils.TestDataset([
            torch.tensor([4, 5, 6, 7], dtype=torch.long),
            torch.tensor([8, 9, 10, 11], dtype=torch.long),
            torch.tensor([12, 13], dtype=torch.long),
        ])
        dataset = MonolingualDataset(dataset, sizes=[4, 4, 2], src_vocab=dictionary)

        config = LanguageModelingConfig(tokens_per_sample=4)
        task = LanguageModelingTask(config, dictionary)

        eval_dataloader = task.eval_lm_dataloader(
            dataset=dataset,
            batch_size=1,
            context_window=2,
        )

        batch = next(eval_dataloader)
        assert batch["net_input"]["src_tokens"][0].tolist() == [4, 5, 6, 7, 1, 1]
        assert batch["target"][0].tolist() == [4, 5, 6, 7, 1, 1]

        batch = next(eval_dataloader)
        assert batch["net_input"]["src_tokens"][0].tolist() == [6, 7, 8, 9, 10, 11]
        assert batch["target"][0].tolist() == [1, 1, 8, 9, 10, 11]

        batch = next(eval_dataloader)
        assert batch["net_input"]["src_tokens"][0].tolist() == [10, 11, 12, 13]
        assert batch["target"][0].tolist() == [1, 1, 12, 13]


if __name__ == "__main__":
    unittest.main()
