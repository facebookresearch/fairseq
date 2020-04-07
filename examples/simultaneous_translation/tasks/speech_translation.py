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
from examples.speech_recognition.tasks.speech_recognition import SpeechRecognitionTask

def get_ast_dataset_from_json(
    data_json_path,
    tgt_dict,
    num_mel_bins,
    online_features,
    mv_norm,
):
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
                [int(i) for i in s[1]["output"]["tokenid"].split(", ")]
            for s in sorted_samples
        ]
        # append eos
        tgt = [[*t, tgt_dict.eos()] for t in tgt]
        return AstDataset(aud_paths, frame_sizes, tgt,tgt_dict, ids, speakers,
                num_mel_bins,
                online_features=online_features,
                mv_norm=mv_norm,
            )


@register_task("speech_translation")
class SpeechTranslationTask(SpeechRecognitionTask):
    """
    Task for training speech recognition model.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--silence-token", default="\u2581",
            help="token for silence (used by w2l)"
        )
        parser.add_argument("--online-features", action="store_true",
                help="Extract features in an online fashion")
        parser.add_argument("--no-mv-norm", action="store_true",
                help="Don't normalize the feature along time dimension")

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        self.tgt_dict = tgt_dict
        self.num_mel_bins = getattr(args, "input_feat_per_channel", 40)
        self.online_features = getattr(args, "online_features", True)
        self.mv_norm = not getattr(args, "no_mv_norm", False)


    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        data_json_path = os.path.join(self.args.data, "{}.json".format(split))
        self.datasets[split] = get_ast_dataset_from_json(
            data_json_path=data_json_path,
            tgt_dict=self.tgt_dict, 
            num_mel_bins=self.num_mel_bins,
            online_features=self.online_features,
            mv_norm=self.mv_norm,
        )
