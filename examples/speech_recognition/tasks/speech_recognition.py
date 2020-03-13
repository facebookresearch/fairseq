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
from examples.speech_recognition.data import AsrDataset
from examples.speech_recognition.data.replabels import replabel_symbol
from examples.speech_recognition.modules.specaugment import SpecAugment


def get_asr_dataset_from_json(data_json_path, tgt_dict):
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
        return AsrDataset(aud_paths, frame_sizes, tgt, tgt_dict, ids, speakers)


@register_task("speech_recognition")
class SpeechRecognitionTask(FairseqTask):
    """
    Task for training speech recognition model.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--silence-token", default="\u2581", help="token for silence (used by w2l)"
        )
        parser.add_argument('--specaugment', action='store_true', default=False,
                            help="If set, SpecAugment is used")
        parser.add_argument('--frequency-masking-pars', type=int, default=13,
                            help="Maximum number of frequencies that can be masked")
        parser.add_argument('--time-masking-pars', type=int, default=13,
                            help="Maximum number of time steps that can be masked")
        parser.add_argument('--frequency-masking-num', type=int, default=2,
                            help="Number of masks to apply along the frequency dimension")
        parser.add_argument('--time-masking-num', type=int, default=2,
                            help="Number of masks to apply along the time dimension")
        parser.add_argument('--specaugment-rate', type=float, default=1.0,
                            help="Probability to apply specaugment to a spectrogram")

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

        specaugment = getattr(args, 'specaugment', False)
        if specaugment:
            self.specaugment = SpecAugment(frequency_masking_pars=args.frequency_masking_pars,
                                           time_masking_pars=args.time_masking_pars,
                                           frequency_masking_num=args.frequency_masking_num,
                                           time_masking_num=args.time_masking_num,
                                           rate=args.specaugment_rate)
        else:
            self.specaugment = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        dict_path = os.path.join(args.data, "dict.txt")
        if not os.path.isfile(dict_path):
            raise FileNotFoundError("Dict not found: {}".format(dict_path))
        tgt_dict = Dictionary.load(dict_path)

        if args.criterion == "ctc_loss":
            tgt_dict.add_symbol("<ctc_blank>")
        elif args.criterion == "asg_loss":
            for i in range(1, args.max_replabel + 1):
                tgt_dict.add_symbol(replabel_symbol(i))

        print("| dictionary: {} types".format(len(tgt_dict)))
        return cls(args, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        data_json_path = os.path.join(self.args.data, "{}.json".format(split))
        self.datasets[split] = get_asr_dataset_from_json(data_json_path, self.tgt_dict)

    def build_generator(self, args):
        w2l_decoder = getattr(args, "w2l_decoder", None)
        if w2l_decoder == "viterbi":
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

            return W2lViterbiDecoder(args, self.target_dictionary)
        elif w2l_decoder == "kenlm":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            return W2lKenLMDecoder(args, self.target_dictionary)
        else:
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

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        if self.specaugment is not None:
            sample = self.specaugment(sample)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output
