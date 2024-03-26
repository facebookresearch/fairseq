# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from argparse import Namespace
from collections import namedtuple
from pathlib import Path

import torch
from tqdm import tqdm

import fairseq
from fairseq import utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.scoring.bleu import SacrebleuScorer
from fairseq.tasks import import_tasks
from tests.speech import TestFairseqSpeech


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestDualInputS2TTransformer(TestFairseqSpeech):
    def setUp(self):
        self.set_up_mustc_de_fbank()

    def import_user_module(self):
        user_dir = (
            Path(fairseq.__file__).parent.parent / "examples/speech_text_joint_to_text"
        )
        Arg = namedtuple("Arg", ["user_dir"])
        arg = Arg(user_dir.__str__())
        utils.import_user_module(arg)

    @torch.no_grad()
    def test_mustc_de_fbank_dualinput_s2t_transformer_checkpoint(self):
        self.import_user_module()
        checkpoint_filename = "checkpoint_ave_10.pt"
        path = self.download(self.base_url, self.root, checkpoint_filename)
        models, cfg, task = load_model_ensemble_and_task(
            [path.as_posix()],
            arg_overrides={
                "data": self.root.as_posix(),
                "config_yaml": "config.yaml",
                "load_pretrain_speech_encoder": "",
                "load_pretrain_text_encoder_last": "",
                "load_pretrain_decoder": "",
                "beam": 10,
                "nbest": 1,
                "lenpen": 1.0,
                "load_speech_only": True,
            },
        )
        if self.use_cuda:
            for model in models:
                model.cuda()
        generator = task.build_generator(models, cfg)
        test_split = "tst-COMMON"
        task.load_dataset(test_split)
        batch_iterator = task.get_batch_iterator(
            dataset=task.dataset(test_split),
            max_tokens=250_000,
            max_positions=(10_000, 1_024),
            num_workers=1,
        ).next_epoch_itr(shuffle=False)

        tokenizer = task.build_tokenizer(cfg.tokenizer)
        bpe = task.build_bpe(cfg.bpe)

        def decode_fn(x):
            if bpe is not None:
                x = bpe.decode(x)
            if tokenizer is not None:
                x = tokenizer.decode(x)
            return x

        scorer_args = {
            "sacrebleu_tokenizer": "13a",
            "sacrebleu_lowercase": False,
            "sacrebleu_char_level": False,
        }
        scorer = SacrebleuScorer(Namespace(**scorer_args))
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
                    print(f"D-{sample_id} {hypo_str}")
                scorer.add_string(tgt_str, hypo_str)
        reference_bleu = 27.3
        result = scorer.result_string()
        print(result + f" (reference: {reference_bleu})")
        res_bleu = float(result.split()[2])
        self.assertAlmostEqual(res_bleu, reference_bleu, delta=0.3)


if __name__ == "__main__":
    unittest.main()
