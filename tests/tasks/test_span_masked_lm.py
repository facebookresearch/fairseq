# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from tempfile import TemporaryDirectory

from fairseq import options
from fairseq.binarizer import FileBinarizer, VocabularyDatasetBinarizer
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks.span_masked_lm import SpanMaskedLMTask
from tests.utils import build_vocab, make_data


class TestSpanMaskedLM(unittest.TestCase):
    def test_masks_token_spans(self):
        with TemporaryDirectory() as dirname:

            # prep input file
            raw_file = os.path.join(dirname, "raw")
            data = make_data(out_file=raw_file)
            vocab = build_vocab(data)

            # binarize
            binarizer = VocabularyDatasetBinarizer(vocab, append_eos=False)
            split = "train"
            bin_file = os.path.join(dirname, split)
            dataset_impl = "mmap"

            FileBinarizer.multiprocess_dataset(
                input_file=raw_file,
                binarizer=binarizer,
                dataset_impl=dataset_impl,
                vocab_size=len(vocab),
                output_prefix=bin_file,
            )

            # adding sentinel tokens
            for i in range(100):
                vocab.add_symbol(f"<extra_id_{i}>")

            # setup task
            train_args = options.parse_args_and_arch(
                options.get_training_parser(),
                [
                    "--task",
                    "span_masked_lm",
                    "--arch",
                    "bart_base",
                    "--seed",
                    "42",
                    dirname,
                ],
            )
            cfg = convert_namespace_to_omegaconf(train_args)
            task = SpanMaskedLMTask(cfg.task, binarizer.dict)

            # load datasets
            original_dataset = task._load_dataset_split(bin_file, 1, False)
            task.load_dataset(split)
            masked_dataset = task.dataset(split)

            iterator = task.get_batch_iterator(
                dataset=masked_dataset,
                max_tokens=65_536,
                max_positions=4_096,
            ).next_epoch_itr(shuffle=False)
            num_tokens = len(vocab)
            for batch in iterator:
                for sample in range(len(batch)):
                    sample_id = batch["id"][sample]
                    original_tokens = original_dataset[sample_id]
                    masked_src_tokens = batch["net_input"]["src_tokens"][sample]
                    masked_src_length = batch["net_input"]["src_lengths"][sample]
                    masked_tgt_tokens = batch["target"][sample]

                    original_offset = 0
                    masked_tgt_offset = 0
                    extra_id_token = len(vocab) - 1
                    for masked_src_token in masked_src_tokens[:masked_src_length]:
                        if masked_src_token == extra_id_token:
                            assert (
                                masked_src_token == masked_tgt_tokens[masked_tgt_offset]
                            )
                            extra_id_token -= 1
                            masked_tgt_offset += 1
                            while (
                                original_offset < len(original_tokens)
                                and masked_tgt_tokens[masked_tgt_offset]
                                != extra_id_token
                            ):
                                assert (
                                    original_tokens[original_offset]
                                    == masked_tgt_tokens[masked_tgt_offset]
                                )
                                original_offset += 1
                                masked_tgt_offset += 1
                        else:
                            assert original_tokens[original_offset] == masked_src_token
                            original_offset += 1


if __name__ == "__main__":
    unittest.main()
