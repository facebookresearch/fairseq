# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random
import sys
from io import StringIO

import torch
import torch.nn.functional as F

import fairseq.distributed.utils as distributed_utils
from fairseq import options, utils
from fairseq.data import Dictionary
from fairseq.data.language_pair_dataset import collate
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.tasks import LegacyFairseqTask
from fairseq_cli import generate, interactive, preprocess, train, validate


def dummy_dictionary(vocab_size, prefix="token_"):
    d = Dictionary()
    for i in range(vocab_size):
        token = prefix + str(i)
        d.add_symbol(token)
    d.finalize(padding_factor=1)  # don't add extra padding symbols
    return d


def dummy_dataloader(
    samples,
    padding_idx=1,
    eos_idx=2,
    batch_size=None,
):
    if batch_size is None:
        batch_size = len(samples)

    # add any missing data to samples
    for i, sample in enumerate(samples):
        if "id" not in sample:
            sample["id"] = i

    # create dataloader
    dataset = TestDataset(samples)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=(lambda samples: collate(samples, padding_idx, eos_idx)),
    )
    return iter(dataloader)


def sequence_generator_setup():
    # construct dummy dictionary
    d = dummy_dictionary(vocab_size=2)

    eos = d.eos()
    w1 = 4
    w2 = 5

    # construct source data
    src_tokens = torch.LongTensor([[w1, w2, eos], [w1, w2, eos]])
    src_lengths = torch.LongTensor([2, 2])

    args = argparse.Namespace()
    unk = 0.0
    args.beam_probs = [
        # step 0:
        torch.FloatTensor(
            [
                # eos      w1   w2
                # sentence 1:
                [0.0, unk, 0.9, 0.1],  # beam 1
                [0.0, unk, 0.9, 0.1],  # beam 2
                # sentence 2:
                [0.0, unk, 0.7, 0.3],
                [0.0, unk, 0.7, 0.3],
            ]
        ),
        # step 1:
        torch.FloatTensor(
            [
                # eos      w1   w2       prefix
                # sentence 1:
                [1.0, unk, 0.0, 0.0],  # w1: 0.9  (emit: w1 <eos>: 0.9*1.0)
                [0.0, unk, 0.9, 0.1],  # w2: 0.1
                # sentence 2:
                [0.25, unk, 0.35, 0.4],  # w1: 0.7  (don't emit: w1 <eos>: 0.7*0.25)
                [0.00, unk, 0.10, 0.9],  # w2: 0.3
            ]
        ),
        # step 2:
        torch.FloatTensor(
            [
                # eos      w1   w2       prefix
                # sentence 1:
                [0.0, unk, 0.1, 0.9],  # w2 w1: 0.1*0.9
                [
                    0.6,
                    unk,
                    0.2,
                    0.2,
                ],  # w2 w2: 0.1*0.1  (emit: w2 w2 <eos>: 0.1*0.1*0.6)
                # sentence 2:
                [
                    0.60,
                    unk,
                    0.4,
                    0.00,
                ],  # w1 w2: 0.7*0.4  (emit: w1 w2 <eos>: 0.7*0.4*0.6)
                [0.01, unk, 0.0, 0.99],  # w2 w2: 0.3*0.9
            ]
        ),
        # step 3:
        torch.FloatTensor(
            [
                # eos      w1   w2       prefix
                # sentence 1:
                [
                    1.0,
                    unk,
                    0.0,
                    0.0,
                ],  # w2 w1 w2: 0.1*0.9*0.9  (emit: w2 w1 w2 <eos>: 0.1*0.9*0.9*1.0)
                [
                    1.0,
                    unk,
                    0.0,
                    0.0,
                ],  # w2 w1 w1: 0.1*0.9*0.1  (emit: w2 w1 w1 <eos>: 0.1*0.9*0.1*1.0)
                # sentence 2:
                [
                    0.1,
                    unk,
                    0.5,
                    0.4,
                ],  # w2 w2 w2: 0.3*0.9*0.99  (emit: w2 w2 w2 <eos>: 0.3*0.9*0.99*0.1)
                [
                    1.0,
                    unk,
                    0.0,
                    0.0,
                ],  # w1 w2 w1: 0.7*0.4*0.4  (emit: w1 w2 w1 <eos>: 0.7*0.4*0.4*1.0)
            ]
        ),
    ]

    task = TestTranslationTask.setup_task(args, d, d)
    model = task.build_model(args)
    tgt_dict = task.target_dictionary

    return tgt_dict, w1, w2, src_tokens, src_lengths, model


def create_dummy_data(data_dir, num_examples=100, maxlen=20, alignment=False):
    def _create_dummy_data(filename):
        data = torch.rand(num_examples * maxlen)
        data = 97 + torch.floor(26 * data).int()
        with open(os.path.join(data_dir, filename), "w") as h:
            offset = 0
            for _ in range(num_examples):
                ex_len = random.randint(1, maxlen)
                ex_str = " ".join(map(chr, data[offset : offset + ex_len]))
                print(ex_str, file=h)
                offset += ex_len

    def _create_dummy_alignment_data(filename_src, filename_tgt, filename):
        with open(os.path.join(data_dir, filename_src), "r") as src_f, open(
            os.path.join(data_dir, filename_tgt), "r"
        ) as tgt_f, open(os.path.join(data_dir, filename), "w") as h:
            for src, tgt in zip(src_f, tgt_f):
                src_len = len(src.split())
                tgt_len = len(tgt.split())
                avg_len = (src_len + tgt_len) // 2
                num_alignments = random.randint(avg_len // 2, 2 * avg_len)
                src_indices = torch.floor(torch.rand(num_alignments) * src_len).int()
                tgt_indices = torch.floor(torch.rand(num_alignments) * tgt_len).int()
                ex_str = " ".join(
                    [
                        "{}-{}".format(src, tgt)
                        for src, tgt in zip(src_indices, tgt_indices)
                    ]
                )
                print(ex_str, file=h)

    _create_dummy_data("train.in")
    _create_dummy_data("train.out")
    _create_dummy_data("valid.in")
    _create_dummy_data("valid.out")
    _create_dummy_data("test.in")
    _create_dummy_data("test.out")

    if alignment:
        _create_dummy_alignment_data("train.in", "train.out", "train.align")
        _create_dummy_alignment_data("valid.in", "valid.out", "valid.align")
        _create_dummy_alignment_data("test.in", "test.out", "test.align")


def preprocess_lm_data(data_dir):
    preprocess_parser = options.get_preprocessing_parser()
    preprocess_args = preprocess_parser.parse_args(
        [
            "--only-source",
            "--trainpref",
            os.path.join(data_dir, "train.out"),
            "--validpref",
            os.path.join(data_dir, "valid.out"),
            "--testpref",
            os.path.join(data_dir, "test.out"),
            "--destdir",
            data_dir,
        ]
    )
    preprocess.main(preprocess_args)


def preprocess_translation_data(data_dir, extra_flags=None):
    preprocess_parser = options.get_preprocessing_parser()
    preprocess_args = preprocess_parser.parse_args(
        [
            "--source-lang",
            "in",
            "--target-lang",
            "out",
            "--trainpref",
            os.path.join(data_dir, "train"),
            "--validpref",
            os.path.join(data_dir, "valid"),
            "--testpref",
            os.path.join(data_dir, "test"),
            "--thresholdtgt",
            "0",
            "--thresholdsrc",
            "0",
            "--destdir",
            data_dir,
        ]
        + (extra_flags or []),
    )
    preprocess.main(preprocess_args)


def preprocess_summarization_data(data_dir, extra_flags=None):
    preprocess_parser = options.get_preprocessing_parser()
    preprocess_args = preprocess_parser.parse_args(
        [
            "--source-lang",
            "in",
            "--target-lang",
            "out",
            "--trainpref",
            os.path.join(data_dir, "train"),
            "--validpref",
            os.path.join(data_dir, "valid"),
            "--testpref",
            os.path.join(data_dir, "test"),
            "--thresholdtgt",
            "0",
            "--thresholdsrc",
            "0",
            "--joined-dictionary",
            "--destdir",
            data_dir,
        ]
        + (extra_flags or []),
    )
    preprocess.main(preprocess_args)


def create_laser_data_and_config_json(data_dir):
    src_langs = ["de", "fr", "ru", "tr", "zh"]
    tgt_langs = ["en", "es"]
    config_json = {}
    config_train_json = []
    src_vocab = None
    tgt_vocab = None

    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            langpair_folder = f"{src_lang}-{tgt_lang}"

            langpair_path = os.path.join(data_dir, langpair_folder)
            os.mkdir(langpair_path)
            create_dummy_data(langpair_path)
            preprocess_translation_data(langpair_path, ["--dataset-impl", "cached"])

            src_vocab = os.path.join(langpair_path, "dict.in.txt")
            tgt_vocab = os.path.join(langpair_path, "dict.out.txt")
            config_train_json.append(
                {
                    "id": 0 if tgt_lang == "en" else 1,
                    "src": os.path.join(langpair_path, "train.in-out.in"),
                    "tgt": os.path.join(langpair_path, "train.in-out.out"),
                }
            )

    config_json["src_vocab"] = src_vocab
    config_json["tgt_vocab"] = tgt_vocab
    config_json["train"] = config_train_json

    with open(os.path.join(data_dir, "laserconfig.json"), "w") as config_file:
        json.dump(config_json, config_file)

    return config_file


def train_translation_model(
    data_dir,
    arch,
    extra_flags=None,
    task="translation",
    run_validation=False,
    lang_flags=None,
    extra_valid_flags=None,
    world_size=1,
):
    if lang_flags is None:
        lang_flags = [
            "--source-lang",
            "in",
            "--target-lang",
            "out",
        ]
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            task,
            data_dir,
            "--save-dir",
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "nag",
            "--lr",
            "0.05",
            "--max-tokens",
            "500",
            "--max-epoch",
            "1",
            "--no-progress-bar",
            "--distributed-world-size",
            str(world_size),
            "--num-workers",
            "0",
        ]
        + lang_flags
        + (extra_flags or []),
    )

    cfg = convert_namespace_to_omegaconf(train_args)
    distributed_utils.call_main(cfg, train.main)

    if run_validation:
        # test validation
        validate_parser = options.get_validation_parser()
        validate_args = options.parse_args_and_arch(
            validate_parser,
            [
                "--task",
                task,
                data_dir,
                "--path",
                os.path.join(data_dir, "checkpoint_last.pt"),
                "--valid-subset",
                "valid",
                "--max-tokens",
                "500",
                "--no-progress-bar",
                "--num-workers",
                "0",
            ]
            + lang_flags
            + (extra_valid_flags or []),
        )
        validate.main(validate_args)


def generate_main(data_dir, extra_flags=None, path=None):
    if extra_flags is None:
        extra_flags = [
            "--print-alignment",
        ]
    if path is None:
        path = os.path.join(data_dir, "checkpoint_last.pt")
    generate_parser = options.get_generation_parser()
    generate_args = options.parse_args_and_arch(
        generate_parser,
        [
            data_dir,
            "--path",
            path,
            "--beam",
            "3",
            "--batch-size",
            "64",
            "--max-len-b",
            "5",
            "--gen-subset",
            "valid",
            "--no-progress-bar",
            "--num-workers",
            "0",
        ]
        + (extra_flags or []),
    )

    # evaluate model in batch mode
    generate.main(generate_args)

    # evaluate model interactively
    generate_args.buffer_size = 0
    generate_args.input = "-"
    generate_args.batch_size = None
    orig_stdin = sys.stdin
    sys.stdin = StringIO("h e l l o\n")
    interactive.main(generate_args)
    sys.stdin = orig_stdin


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.sizes = None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestTranslationTask(LegacyFairseqTask):
    def __init__(self, args, src_dict, tgt_dict, model):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.model = model

    @classmethod
    def setup_task(cls, args, src_dict=None, tgt_dict=None, model=None):
        return cls(args, src_dict, tgt_dict, model)

    def build_model(self, args):
        return TestModel.build_model(args, self)

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict


class TestModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        encoder = TestEncoder(args, task.source_dictionary)
        decoder = TestIncrementalDecoder(args, task.target_dictionary)
        return cls(encoder, decoder)


class TestEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        return EncoderOut(
            encoder_out=src_tokens,
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        return EncoderOut(
            encoder_out=encoder_out.encoder_out.index_select(0, new_order),
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )


class TestIncrementalDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        assert hasattr(args, "beam_probs") or hasattr(args, "probs")
        args.max_decoder_positions = getattr(args, "max_decoder_positions", 100)
        self.args = args

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bbsz = prev_output_tokens.size(0)
        vocab = len(self.dictionary)
        src_len = encoder_out.encoder_out.size(1)
        tgt_len = prev_output_tokens.size(1)

        # determine number of steps
        if incremental_state is not None:
            # cache step number
            step = utils.get_incremental_state(self, incremental_state, "step")
            if step is None:
                step = 0
            utils.set_incremental_state(self, incremental_state, "step", step + 1)
            steps = [step]
        else:
            steps = list(range(tgt_len))

        # define output in terms of raw probs
        if hasattr(self.args, "probs"):
            assert (
                self.args.probs.dim() == 3
            ), "expected probs to have size bsz*steps*vocab"
            probs = self.args.probs.index_select(1, torch.LongTensor(steps))
        else:
            probs = torch.FloatTensor(bbsz, len(steps), vocab).zero_()
            for i, step in enumerate(steps):
                # args.beam_probs gives the probability for every vocab element,
                # starting with eos, then unknown, and then the rest of the vocab
                if step < len(self.args.beam_probs):
                    probs[:, i, self.dictionary.eos() :] = self.args.beam_probs[step]
                else:
                    probs[:, i, self.dictionary.eos()] = 1.0

        # random attention
        attn = torch.rand(bbsz, tgt_len, src_len)

        dev = prev_output_tokens.device
        return probs.to(dev), {"attn": [attn.to(dev)]}

    def get_normalized_probs(self, net_output, log_probs, _):
        # the decoder returns probabilities directly
        probs = net_output[0]
        if log_probs:
            return probs.log()
        else:
            return probs

    def max_positions(self):
        return self.args.max_decoder_positions


class TestReshapingEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        b_sz, t_sz = src_tokens.shape
        padding_needed = t_sz % 2
        x = src_tokens
        if padding_needed > 0:
            padding_needed = 2 - padding_needed
            x = F.pad(x, (0, padding_needed))

        return EncoderOut(
            encoder_out=x.view(b_sz, -1, 2),
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        return EncoderOut(
            encoder_out=encoder_out.encoder_out.index_select(0, new_order),
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )


class TestReshapingModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        encoder = TestReshapingEncoder(args, task.source_dictionary)
        decoder = TestIncrementalDecoder(args, task.target_dictionary)
        return cls(encoder, decoder)


class TestAdditionalInputEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        assert "fancy_other_input" in kwargs
        assert kwargs["fancy_other_input"] is not None
        return EncoderOut(
            encoder_out=src_tokens,
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        return EncoderOut(
            encoder_out=encoder_out.encoder_out.index_select(0, new_order),
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )


class TestAdditionalInputModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        encoder = TestAdditionalInputEncoder(args, task.source_dictionary)
        decoder = TestIncrementalDecoder(args, task.target_dictionary)
        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out


def train_language_model(
    data_dir,
    arch,
    extra_flags=None,
    run_validation=False,
    extra_valid_flags=None,
    task="language_modeling",
    world_size=1,
):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            task,
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--max-tokens",
            "500",
            "--tokens-per-sample",
            "500",
            "--save-dir",
            data_dir,
            "--max-epoch",
            "1",
            "--no-progress-bar",
            "--distributed-world-size",
            str(world_size),
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
        ]
        + (extra_flags or []),
    )
    cfg = convert_namespace_to_omegaconf(train_args)
    distributed_utils.call_main(cfg, train.main)

    if run_validation:
        # test validation
        validate_parser = options.get_validation_parser()
        validate_args = options.parse_args_and_arch(
            validate_parser,
            [
                "--task",
                task,
                data_dir,
                "--path",
                os.path.join(data_dir, "checkpoint_last.pt"),
                "--valid-subset",
                "valid",
                "--max-tokens",
                "500",
                "--no-progress-bar",
                "--num-workers",
                "0",
            ]
            + (extra_valid_flags or []),
        )
        validate.main(validate_args)
