# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from tqdm import tqdm

from fairseq import utils
from tests.speech import TestFairseqSpeech


class TestXMTransformer(TestFairseqSpeech):
    def setUp(self):
        self.set_up_sotasty_es_en()

    @torch.no_grad()
    def test_sotasty_es_en_600m_checkpoint(self):
        models, cfg, task, generator = self.download_and_load_checkpoint(
            "xm_transformer_600m_es_en_md.pt",
            arg_overrides={"config_yaml": "cfg_es_en.yaml"},
        )
        if not self.use_cuda:
            return

        batch_iterator = self.get_batch_iterator(
            task, "sotasty_es_en_test_ted", 3_000_000, (1_000_000, 1_024)
        )
        scorer = self.get_bleu_scorer()
        progress = tqdm(enumerate(batch_iterator), total=len(batch_iterator))
        for batch_idx, sample in progress:
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample
            hypo = task.inference_step(generator, models, sample)
            for i, sample_id in enumerate(sample["id"].tolist()):
                tgt_tokens = (
                    utils.strip_pad(sample["target"][i, :], task.tgt_dict.pad())
                    .int()
                    .cpu()
                )
                tgt_str = task.tgt_dict.string(tgt_tokens, "sentencepiece")
                hypo_str = task.tgt_dict.string(
                    hypo[i][0]["tokens"].int().cpu(), "sentencepiece"
                )
                if batch_idx == 0 and i < 3:
                    print(f"T-{sample_id} {tgt_str}")
                    print(f"H-{sample_id} {hypo_str}")
                scorer.add_string(tgt_str, hypo_str)
        reference_bleu = 31.7
        print(f"{scorer.result_string()} (reference: {reference_bleu})")
        self.assertAlmostEqual(scorer.score(), reference_bleu, delta=0.2)


if __name__ == "__main__":
    unittest.main()
