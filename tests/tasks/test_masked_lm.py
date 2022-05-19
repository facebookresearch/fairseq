# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from tempfile import TemporaryDirectory

from fairseq.binarizer import FileBinarizer, VocabularyDatasetBinarizer
from fairseq.tasks.masked_lm import MaskedLMConfig, MaskedLMTask
from tests.utils import build_vocab, make_data


class TestMaskedLM(unittest.TestCase):
    def test_masks_tokens(self):
        with TemporaryDirectory() as dirname:

            # prep input file
            raw_file = os.path.join(dirname, "raw")
            data = make_data(out_file=raw_file)
            vocab = build_vocab(data)

            # binarize
            binarizer = VocabularyDatasetBinarizer(vocab, append_eos=False)
            split = "train"
            bin_file = os.path.join(dirname, split)
            FileBinarizer.multiprocess_dataset(
                input_file=raw_file,
                binarizer=binarizer,
                dataset_impl="mmap",
                vocab_size=len(vocab),
                output_prefix=bin_file,
            )

            # setup task
            cfg = MaskedLMConfig(
                data=dirname,
                seed=42,
                mask_prob=0.5,  # increasing the odds of masking
                random_token_prob=0,  # avoiding random tokens for exact match
                leave_unmasked_prob=0,  # always masking for exact match
            )
            task = MaskedLMTask(cfg, binarizer.dict)

            original_dataset = task._load_dataset_split(bin_file, 1, False)

            # load datasets
            task.load_dataset(split)
            masked_dataset = task.dataset(split)

            mask_index = task.source_dictionary.index("<mask>")
            iterator = task.get_batch_iterator(
                dataset=masked_dataset,
                max_tokens=65_536,
                max_positions=4_096,
            ).next_epoch_itr(shuffle=False)
            for batch in iterator:
                for sample in range(len(batch)):
                    net_input = batch["net_input"]
                    masked_src_tokens = net_input["src_tokens"][sample]
                    masked_src_length = net_input["src_lengths"][sample]
                    masked_tgt_tokens = batch["target"][sample]

                    sample_id = batch["id"][sample]
                    original_tokens = original_dataset[sample_id]
                    original_tokens = original_tokens.masked_select(
                        masked_src_tokens[:masked_src_length] == mask_index
                    )
                    masked_tokens = masked_tgt_tokens.masked_select(
                        masked_tgt_tokens != task.source_dictionary.pad()
                    )

                    assert masked_tokens.equal(original_tokens)


if __name__ == "__main__":
    unittest.main()
