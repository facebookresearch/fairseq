# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import os

from fairseq.data import (
    BacktranslationDataset,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    NoisingDataset,
    RoundRobinZipDatasets,
)
from fairseq.models import FairseqMultiModel
from fairseq.sequence_generator import SequenceGenerator


from .multilingual_translation import MultilingualTranslationTask
from . import register_task


def _get_bt_dataset_key(lang_pair):
    return "bt:" + lang_pair


def _get_denoising_dataset_key(lang_pair):
    return "denoising:" + lang_pair


# ported from UnsupervisedMT
def parse_lambda_config(x):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                             # to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                             # iterations, then will linearly increase to 1 until iteration 2000
    """
    split = x.split(',')
    if len(split) == 1:
        return float(x), None
    else:
        split = [s.split(':') for s in split]
        assert all(len(s) == 2 for s in split)
        assert all(k.isdigit() for k, _ in split)
        assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
        return float(split[0][1]), [(int(k), float(v)) for k, v in split]


@register_task('semisupervised_translation')
class SemisupervisedTranslationTask(MultilingualTranslationTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, instead of `--lang-pairs`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        MultilingualTranslationTask.add_args(parser)
        parser.add_argument('--lambda-parallel-config', default="1.0", type=str, metavar='CONFIG',
                            help='cross-entropy reconstruction coefficient (parallel data). '
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--lambda-denoising-config', default="0.0", type=str, metavar='CONFIG',
                            help='Cross-entropy reconstruction coefficient (denoising autoencoding)'
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--lambda-otf-bt-config', default="0.0", type=str, metavar='CONFIG',
                            help='cross-entropy reconstruction coefficient (on-the-fly back-translation parallel data)'
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--bt-max-len-a', default=1.1, type=float, metavar='N',
                            help='generate back-translated sequences of maximum length ax + b, where x is the '
                                 'source length')
        parser.add_argument('--bt-max-len-b', default=10.0, type=float, metavar='N',
                            help='generate back-translated sequences of maximum length ax + b, where x is the '
                                 'source length')
        parser.add_argument('--bt-beam-size', default=1, type=int, metavar='N',
                            help='beam size used in beam search of online back-translation')
        parser.add_argument('--max-word-shuffle-distance', default=3.0, type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.2, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.lambda_parallel, self.lambda_parallel_steps = parse_lambda_config(args.lambda_parallel_config)
        self.lambda_otf_bt, self.lambda_otf_bt_steps = parse_lambda_config(args.lambda_otf_bt_config)
        self.lambda_denoising, self.lambda_denoising_steps = parse_lambda_config(args.lambda_denoising_config)
        if (self.lambda_denoising > 0.0 or self.lambda_denoising_steps is not None):
            denoising_lang_pairs = [
                "%s-%s" % (tgt, tgt)
                for tgt in {lang_pair.split('-')[1] for lang_pair in args.lang_pairs}
            ]
            self.model_lang_pairs += denoising_lang_pairs
        self.backtranslate_datasets = {}
        self.backtranslators = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        dicts, training = MultilingualTranslationTask.prepare(args, **kwargs)
        return cls(args, dicts, training)

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a dataset split."""

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        def split_exists(split, src, tgt, lang):
            if src is not None:
                filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            else:
                filename = os.path.join(data_path, '{}.{}-None.{}'.format(split, src, tgt))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        # load parallel datasets
        src_datasets, tgt_datasets = {}, {}
        if (self.lambda_parallel > 0.0 or self.lambda_parallel_steps is not None or not split.startswith("train")):
            for lang_pair in self.args.lang_pairs:
                src, tgt = lang_pair.split('-')
                if split_exists(split, src, tgt, src):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, src, tgt))
                elif split_exists(split, tgt, src, src):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, tgt, src))
                else:
                    continue
                src_datasets[lang_pair] = indexed_dataset(prefix + src, self.dicts[src])
                tgt_datasets[lang_pair] = indexed_dataset(prefix + tgt, self.dicts[tgt])
                print('| parallel-{} {} {} examples'.format(data_path, split, len(src_datasets[lang_pair])))
            if len(src_datasets) == 0:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        # back translation datasets
        backtranslate_datasets = {}
        if (self.lambda_otf_bt > 0.0 or self.lambda_otf_bt_steps is not None) and split.startswith("train"):
            for lang_pair in self.args.lang_pairs:
                src, tgt = lang_pair.split('-')
                if not split_exists(split, tgt, None, tgt):
                    raise FileNotFoundError('Dataset not found: backtranslation {} ({})'.format(split, data_path))
                filename = os.path.join(data_path, '{}.{}-None.{}'.format(split, tgt, tgt))
                dataset = indexed_dataset(filename, self.dicts[tgt])
                lang_pair_dataset_tgt = LanguagePairDataset(
                    dataset,
                    dataset.sizes,
                    self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                )
                lang_pair_dataset = LanguagePairDataset(
                    dataset,
                    dataset.sizes,
                    src_dict=self.dicts[src],
                    tgt=dataset,
                    tgt_sizes=dataset.sizes,
                    tgt_dict=self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                )
                backtranslate_datasets[lang_pair] = BacktranslationDataset(
                    tgt_dataset=self.alter_dataset_langtok(
                        lang_pair_dataset_tgt,
                        src_eos=self.dicts[tgt].eos(),
                        src_lang=tgt,
                        tgt_lang=src,
                    ),
                    backtranslation_fn=self.backtranslators[lang_pair],
                    src_dict=self.dicts[src], tgt_dict=self.dicts[tgt],
                    output_collater=self.alter_dataset_langtok(
                        lang_pair_dataset=lang_pair_dataset,
                        src_eos=self.dicts[src].eos(),
                        src_lang=src,
                        tgt_eos=self.dicts[tgt].eos(),
                        tgt_lang=tgt,
                    ).collater,
                )
                print('| backtranslate-{}: {} {} {} examples'.format(
                    tgt, data_path, split, len(backtranslate_datasets[lang_pair]),
                ))
                self.backtranslate_datasets[lang_pair] = backtranslate_datasets[lang_pair]

        # denoising autoencoder
        noising_datasets = {}
        if (self.lambda_denoising > 0.0 or self.lambda_denoising_steps is not None) and split.startswith("train"):
            for lang_pair in self.args.lang_pairs:
                _, tgt = lang_pair.split('-')
                if not split_exists(split, tgt, None, tgt):
                    continue
                filename = os.path.join(data_path, '{}.{}-None.{}'.format(split, tgt, tgt))
                tgt_dataset1 = indexed_dataset(filename, self.dicts[tgt])
                tgt_dataset2 = indexed_dataset(filename, self.dicts[tgt])
                noising_dataset = NoisingDataset(
                    tgt_dataset1,
                    self.dicts[tgt],
                    seed=1,
                    max_word_shuffle_distance=self.args.max_word_shuffle_distance,
                    word_dropout_prob=self.args.word_dropout_prob,
                    word_blanking_prob=self.args.word_blanking_prob,
                )
                noising_datasets[lang_pair] = self.alter_dataset_langtok(
                    LanguagePairDataset(
                        noising_dataset,
                        tgt_dataset1.sizes,
                        self.dicts[tgt],
                        tgt_dataset2,
                        tgt_dataset2.sizes,
                        self.dicts[tgt],
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_target,
                    ),
                    src_eos=self.dicts[tgt].eos(),
                    src_lang=tgt,
                    tgt_eos=self.dicts[tgt].eos(),
                    tgt_lang=tgt,
                )
                print('| denoising-{}: {} {} {} examples'.format(
                    tgt, data_path, split, len(noising_datasets[lang_pair]),
                ))

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            src_dataset, tgt_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            return self.alter_dataset_langtok(
                LanguagePairDataset(
                    src_dataset, src_dataset.sizes, self.dicts[src],
                    tgt_dataset, tgt_dataset.sizes, self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                ),
                self.dicts[src].eos(),
                src,
                self.dicts[tgt].eos(),
                tgt,
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in src_datasets.keys()
            ] + [
                (_get_bt_dataset_key(lang_pair), dataset)
                for lang_pair, dataset in backtranslate_datasets.items()
            ] + [
                (_get_denoising_dataset_key(lang_pair), dataset)
                for lang_pair, dataset in noising_datasets.items()
            ]),
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiModel):
            raise ValueError('SemisupervisedTranslationTask requires a FairseqMultiModel architecture')

        # create SequenceGenerator for each model that has backtranslation dependency on it
        self.sequence_generators = {}
        if (self.lambda_otf_bt > 0.0 or self.lambda_otf_bt_steps is not None) and self.training:
            for lang_pair in self.args.lang_pairs:
                src, tgt = lang_pair.split('-')
                key = '{}-{}'.format(tgt, src)
                self.sequence_generators[key] = SequenceGenerator(
                    tgt_dict=self.dicts[src],
                    beam_size=args.bt_beam_size,
                    max_len_a=args.bt_max_len_a,
                    max_len_b=args.bt_max_len_b,
                )
                decoder_lang_tok_idx = self.get_decoder_langtok(src)

                def backtranslate_fn(
                    sample, model=model.models[key],
                    bos_token=decoder_lang_tok_idx,
                    sequence_generator=self.sequence_generators[key],
                ):
                    return sequence_generator.generate(
                        [model],
                        sample,
                        bos_token=bos_token,
                    )
                self.backtranslators[lang_pair] = backtranslate_fn

        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        def forward_backward(model, samples, logging_output_key, weight):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output[logging_output_key] = logging_output

        if self.lambda_denoising > 0.0:
            for lang_pair in self.args.lang_pairs:
                _, tgt = lang_pair.split('-')
                sample_key = _get_denoising_dataset_key(lang_pair)
                forward_backward(model.models['{0}-{0}'.format(tgt)], sample[sample_key], sample_key, self.lambda_denoising)

        if self.lambda_parallel > 0.0:
            for lang_pair in self.args.lang_pairs:
                forward_backward(model.models[lang_pair], sample[lang_pair], lang_pair, self.lambda_parallel)

        if self.lambda_otf_bt > 0.0:
            for lang_pair in self.args.lang_pairs:
                sample_key = _get_bt_dataset_key(lang_pair)
                forward_backward(model.models[lang_pair], sample[sample_key], sample_key, self.lambda_otf_bt)

        return agg_loss, agg_sample_size, agg_logging_output

    def update_step(self, num_updates):
        def lambda_step_func(config, n_iter):
            """
            Update a lambda value according to its schedule configuration.
            """
            ranges = [i for i in range(len(config) - 1) if config[i][0] <= n_iter < config[i + 1][0]]
            if len(ranges) == 0:
                assert n_iter >= config[-1][0]
                return config[-1][1]
            assert len(ranges) == 1
            i = ranges[0]
            x_a, y_a = config[i]
            x_b, y_b = config[i + 1]
            return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)

        if self.lambda_parallel_steps is not None:
            self.lambda_parallel = lambda_step_func(self.lambda_parallel_steps, num_updates)
        if self.lambda_denoising_steps is not None:
            self.lambda_denoising = lambda_step_func(self.lambda_denoising_steps, num_updates)
        if self.lambda_otf_bt_steps is not None:
            self.lambda_otf_bt = lambda_step_func(self.lambda_otf_bt_steps, num_updates)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        # aggregate logging outputs for each language pair
        logging_output_keys = {
            key
            for logging_output in logging_outputs
            for key in logging_output
        }
        lang_pair_keys = set(self.args.lang_pairs + [
            _get_bt_dataset_key(lang_pair)
            for lang_pair in self.args.lang_pairs
        ] + [
            _get_denoising_dataset_key(lang_pair)
            for lang_pair in self.args.lang_pairs
        ])
        logging_output_keys = logging_output_keys.intersection(lang_pair_keys)
        return super().aggregate_logging_outputs(logging_outputs, criterion, logging_output_keys)
