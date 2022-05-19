# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from collections import OrderedDict

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from ..utils import set_seed


class MMDataset(Dataset):
    """
    A generic multi-modal dataset.
        Args:
            `meta_processor`: a meta processor,
                handling loading meta data and return video_id and text_id.
            `video_processor`: a video processor,
                handling e.g., decoding, loading .np files.
            `text_processor`: a text processor,
                handling e.g., tokenization.
            `aligner`: combine the video and text feature
                as one training example.
    """

    def __init__(
        self,
        meta_processor,
        video_processor,
        text_processor,
        align_processor,
    ):
        self.split = meta_processor.split
        self.meta_processor = meta_processor
        self.video_processor = video_processor
        self.text_processor = text_processor
        self.align_processor = align_processor

    def __len__(self):
        return len(self.meta_processor)

    def __getitem__(self, idx):
        if self.split == "test":
            set_seed(idx)
        video_id, text_id = self.meta_processor[idx]
        video_feature = self.video_processor(video_id)
        text_feature = self.text_processor(text_id)
        output = self.align_processor(video_id, video_feature, text_feature)
        # TODO (huxu): the following is for debug purpose.
        output.update({"idx": idx})
        return output

    def collater(self, samples):
        """This collator is deprecated.
        set self.collator = MMDataset.collater.
        see collator in FairseqMMDataset.
        """

        if len(samples) == 0:
            return {}
        if isinstance(samples[0], dict):
            batch = OrderedDict()
            for key in samples[0]:
                if samples[0][key] is not None:
                    batch[key] = default_collate(
                        [sample[key] for sample in samples])
                # if torch.is_tensor(batch[key]):
                #    print(key, batch[key].size())
                # else:
                #    print(key, len(batch[key]))
            return batch
        else:
            return default_collate(samples)

    def print_example(self, output):
        print("[one example]", output["video_id"])
        if (
            hasattr(self.align_processor, "subsampling")
            and self.align_processor.subsampling is not None
            and self.align_processor.subsampling > 1
        ):
            for key in output:
                if torch.is_tensor(output[key]):
                    output[key] = output[key][0]

        # search tokenizer to translate ids back.
        tokenizer = None
        if hasattr(self.text_processor, "tokenizer"):
            tokenizer = self.text_processor.tokenizer
        elif hasattr(self.align_processor, "tokenizer"):
            tokenizer = self.align_processor.tokenizer
        if tokenizer is not None:
            caps = output["caps"].tolist()
            if isinstance(caps[0], list):
                caps = caps[0]
            print("caps", tokenizer.decode(caps))
            print("caps", tokenizer.convert_ids_to_tokens(caps))

        for key, value in output.items():
            if torch.is_tensor(value):
                if len(value.size()) >= 3:  # attention_mask.
                    print(key, value.size())
                    print(key, "first", value[0, :, :])
                    print(key, "last", value[-1, :, :])
                else:
                    print(key, value)
        print("[end of one example]")
