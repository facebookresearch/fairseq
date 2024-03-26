# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from tqdm import tqdm

from fairseq import utils
from fairseq.tasks.text_to_speech import batch_mel_cepstral_distortion
from tests.speech import TestFairseqSpeech


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestFastSpeech2(TestFairseqSpeech):
    def setUp(self):
        self.set_up_ljspeech()

    @torch.no_grad()
    def test_ljspeech_fastspeech2_checkpoint(self):
        models, cfg, task, generator = self.download_and_load_checkpoint(
            "ljspeech_fastspeech2_g2p.pt",
            arg_overrides={
                "config_yaml": "cfg_ljspeech_g2p.yaml",
                "vocoder": "griffin_lim",
                "fp16": False,
            },
        )

        batch_iterator = self.get_batch_iterator(task, "ljspeech_test", 65_536, 4_096)
        progress = tqdm(batch_iterator, total=len(batch_iterator))
        mcd, n_samples = 0.0, 0
        for sample in progress:
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample
            hypos = generator.generate(models[0], sample, has_targ=True)
            rets = batch_mel_cepstral_distortion(
                [hypo["targ_waveform"] for hypo in hypos],
                [hypo["waveform"] for hypo in hypos],
                sr=task.sr,
            )
            mcd += sum(d.item() for d, _ in rets)
            n_samples += len(sample["id"].tolist())

        mcd = round(mcd / n_samples, 1)
        reference_mcd = 3.2
        print(f"MCD: {mcd} (reference: {reference_mcd})")
        self.assertAlmostEqual(mcd, reference_mcd, delta=0.1)


if __name__ == "__main__":
    unittest.main()
