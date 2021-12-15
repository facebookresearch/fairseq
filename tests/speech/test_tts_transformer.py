# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from tqdm import tqdm

from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq import utils
from fairseq.tasks.text_to_speech import batch_mel_cepstral_distortion
from tests.speech import TestFairseqSpeech


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestTTSTransformer(TestFairseqSpeech):
    def setUp(self):
        self.set_up_ljspeech()

    @torch.no_grad()
    def test_ljspeech_tts_transformer_checkpoint(self):
        checkpoint_filename = "ljspeech_transformer_g2p.pt"
        path = self.download(self.base_url, self.root, checkpoint_filename)

        models, cfg, task = load_model_ensemble_and_task(
            [path.as_posix()], arg_overrides={
                "data": self.root.as_posix(),
                "config_yaml": "cfg_ljspeech_g2p.yaml",
                "vocoder": "griffin_lim",
                "fp16": False
            }
        )
        if self.use_cuda:
            for model in models:
                model.cuda()

        test_split = "ljspeech_test"
        task.load_dataset(test_split)
        batch_iterator = task.get_batch_iterator(
            dataset=task.dataset(test_split),
            max_tokens=65_536, max_positions=768, num_workers=1
        ).next_epoch_itr(shuffle=False)
        progress = tqdm(batch_iterator, total=len(batch_iterator))
        generator = task.build_generator(models, cfg)

        mcd, n_samples = 0., 0
        for sample in progress:
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample
            hypos = generator.generate(models[0], sample, has_targ=True)
            rets = batch_mel_cepstral_distortion(
                [hypo["targ_waveform"] for hypo in hypos],
                [hypo["waveform"] for hypo in hypos],
                sr=task.sr
            )
            mcd += sum(d.item() for d, _ in rets)
            n_samples += len(sample["id"].tolist())

        mcd = round(mcd / n_samples, 1)
        reference_mcd = 3.3
        print(f"MCD: {mcd} (reference: {reference_mcd})")
        self.assertAlmostEqual(mcd, reference_mcd, delta=0.1)


if __name__ == "__main__":
    unittest.main()
