# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch

from tests.test_train import mock_dict
import tests.utils as test_utils

from fairseq.data.audio.multi_modality_dataset import LangPairMaskDataset
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.data.transform_eos_lang_pair_dataset import TransformEosLangPairDataset



class TestMultiModality(unittest.TestCase):
    def test_langpair_mask_collater(self):
        fake_src_dataset = test_utils.TestDataset(
            [
                torch.tensor([2, 5, 6, 7], dtype=torch.long),
                torch.tensor([2, 9, 10, 11], dtype=torch.long),
                torch.tensor([2, 13], dtype=torch.long),
            ]
        )
        fake_tgt_dataset = test_utils.TestDataset(
            [
                torch.tensor([2, 7, 6, 5], dtype=torch.long),
                torch.tensor([2, 11, 10, 9], dtype=torch.long),
                torch.tensor([2, 12], dtype=torch.long),
            ]
        )
        lp_dataset = LanguagePairDataset(
            fake_src_dataset,
            fake_src_dataset.sizes,
            mock_dict(),
            fake_tgt_dataset,
            fake_tgt_dataset.sizes,
            mock_dict()
        )
        te_dataset = TransformEosLangPairDataset(
            lp_dataset,
            src_eos=2,
            tgt_bos=2,
            new_tgt_bos=20
        )
        lm_dataset = LangPairMaskDataset(
            te_dataset,
            src_bos=2,
            src_eos=2,
        )
        samples = lm_dataset.collater([lm_dataset[0], lm_dataset[1], lm_dataset[2]])
        self.assertTrue(samples['net_input']['prev_output_tokens'][0][0].item() == 20)        
        

if __name__ == "__main__":
    unittest.main()