# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re

import torch
from fairseq.data import Dictionary
from fairseq.tasks import FairseqTask, register_task
from examples.simultaneous_translation.data import AstDataset

def get_ast_dataset_from_json(data_json_path, tgt_dict, num_mel_bins=80):
    """
    Parse data json and create dataset.
    See scripts/asr_prep_json.py which pack json from raw files

    Json example:
    {
    "utts": {
        "4771-29403-0025": {
            "input": {
                "length_ms": 170,
                "path": "/tmp/file1.flac"
            },
            "output": {
                "text": "HELLO \n",
                "token": "HE LLO",
                "tokenid": "4815, 861"
            }
        },
        "1564-142299-0096": {
            ...
        }
    }
    """
    if not os.path.isfile(data_json_path):
        raise FileNotFoundError("Dataset not found: {}".format(data_json_path))
    with open(data_json_path, "rb") as f:
        data_samples = json.load(f)["utts"]
        assert len(data_samples) != 0
        sorted_samples = sorted(
            data_samples.items(),
            key=lambda sample: int(sample[1]["input"]["length_ms"]),
            reverse=True,
        )
        aud_paths = [s[1]["input"]["path"] for s in sorted_samples]
        ids = [s[0] for s in sorted_samples]
        speakers = []
        for s in sorted_samples:
            m = re.search("(.+?)-(.+?)-(.+?)", s[0])
            speakers.append(m.group(1) + "_" + m.group(2))
        frame_sizes = [s[1]["input"]["length_ms"] for s in sorted_samples]
        tgt = [
            torch.LongTensor([int(i) for i in s[1]["output"]["tokenid"].split(", ")])
            for s in sorted_samples
        ]
        # append eos
        tgt = [torch.cat([t, torch.LongTensor([tgt_dict.eos()])]) for t in tgt]
        return AstDataset(aud_paths, frame_sizes, tgt, tgt_dict, ids, speakers, num_mel_bins)


@register_task("speech_translation")
class SpeechTranslationTask(FairseqTask):
    """
    Task for training speech recognition model.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--silence-#token", default="\u2581", help="token for silence (used by w2l)"
        )

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.num_mel_bins = getattr(args, "input_feat_per_channel", 80)
    
    def set_num_mel_bins(self, input_feat_per_channel):
        self.num_mel_bins = input_feat_per_channel

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        dict_path = os.path.join(args.data, "dict.txt")
        if not os.path.isfile(dict_path):
            raise FileNotFoundError("Dict not found: {}".format(dict_path))
        tgt_dict = Dictionary.load(dict_path)

        print("| dictionary: {} types".format(len(tgt_dict)))
        return cls(args, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        data_json_path = os.path.join(self.args.data, "{}.json".format(split))
        self.datasets[split] = get_ast_dataset_from_json(
            data_json_path,
            self.tgt_dict, 
            self.num_mel_bins
        )

    def build_generator(self, args):
        return super().build_generator(args)

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.tgt_dict

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None
