# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from argparse import Namespace

import torch
from tqdm import tqdm

from fairseq import utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.scoring.wer import WerScorer
from tests.speech import TestFairseqSpeech


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestS2TTransformer(TestFairseqSpeech):
    def setUp(self):
        self.set_up_librispeech()

    @torch.no_grad()
    def test_librispeech_s2t_transformer_s_checkpoint(self):
        checkpoint_filename = "librispeech_transformer_s.pt"
        path = self.download(self.base_url, self.root, checkpoint_filename)

        models, cfg, task = load_model_ensemble_and_task(
            [path.as_posix()],
            arg_overrides={
                "data": self.root.as_posix(),
                "config_yaml": "cfg_librispeech.yaml",
            },
        )
        if self.use_cuda:
            for model in models:
                model.cuda()
        generator = task.build_generator(models, cfg)
        test_split = "librispeech_test-other"
        task.load_dataset(test_split)
        batch_iterator = task.get_batch_iterator(
            dataset=task.dataset(test_split),
            max_tokens=65_536,
            max_positions=(4_096, 1_024),
            num_workers=1,
        ).next_epoch_itr(shuffle=False)

        scorer_args = {
            "wer_tokenizer": "none",
            "wer_lowercase": False,
            "wer_remove_punct": False,
            "wer_char_level": False,
        }
        scorer = WerScorer(Namespace(**scorer_args))
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
        reference_wer = 9.0
        print(scorer.result_string() + f" (reference: {reference_wer})")
        self.assertAlmostEqual(scorer.score(), reference_wer, delta=0.3)


if __name__ == "__main__":
    unittest.main()
