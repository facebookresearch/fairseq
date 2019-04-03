# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import copy
import os

import torch

from fairseq import options
from fairseq.data import (
    BacktranslationDataset,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    NoisingDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
)
from fairseq.models import FairseqMultiModel
from fairseq.sequence_generator import SequenceGenerator


from . import FairseqTask, register_task


def _get_bt_dataset_key(lang_pair):
    return "bt:" + lang_pair


def _get_denoising_dataset_key(lang_pair):
    return "denoising:" + lang_pair


def _lang_token(lang):
    return "__%s__" % lang


def _lang_token_index(dic, lang):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        f'cannot find language token for lang {lang}'
    return idx


# ported from UnsupervisedMT
def parse_lambda_config(x):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000 iterations, then will linearly increase to 1 until iteration 2000
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


@register_task('multilingual_translation')
class MultilingualTranslationTask(FairseqTask):
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
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target language token. (src/tgt)')
        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')
        parser.add_argument('--lambda-parallel-config', default="1.0", type=str, metavar='F',
                            help='cross-entropy reconstruction coefficient (parallel data)')
        parser.add_argument('--lambda-denoising-config', default="0.0", type=str, metavar='CONFIG',
                            help='Cross-entropy reconstruction coefficient (denoising autoencoding)')
        parser.add_argument('--lambda-otf-bt-config', default="0.0", type=str, metavar='F',
                            help='cross-entropy reconstruction coefficient (on-the-fly back-translation parallel data)')
        parser.add_argument('--bt-max-len-a', default=1.1, type=float, metavar='N',
                            help='generate back-translated sequences of maximum length ax + b, where x is the source length')
        parser.add_argument('--bt-max-len-b', default=10.0, type=float, metavar='N',
                            help='generate back-translated sequences of maximum length ax + b, where x is the source length')
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
        super().__init__(args)
        self.dicts = dicts
        self.lang_pairs = args.lang_pairs
        self.langs = list(dicts.keys())
        self.training = training
        self.model_lang_pairs = copy.copy(args.lang_pairs)
        self.lambda_parallel, self.lambda_parallel_steps = parse_lambda_config(args.lambda_parallel_config)
        self.lambda_otf_bt, self.lambda_otf_bt_steps = parse_lambda_config(args.lambda_otf_bt_config)
        self.lambda_denoising, self.lambda_denoising_steps = parse_lambda_config(args.lambda_denoising_config)
        if (self.lambda_denoising > 0.0 or self.lambda_denoising_steps is not None):
            denoising_lang_pairs = [
                "%s-%s" % (tgt, tgt)
                for tgt in set([lang_pair.split('-')[1] for lang_pair in args.lang_pairs])
            ]
            self.model_lang_pairs += denoising_lang_pairs
        self.backtranslate_datasets = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        args.lang_pairs = args.lang_pairs.split(',')
        sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        if args.source_lang is not None or args.target_lang is not None:
            training = False
            args.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        else:
            training = True
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')

        # load dictionaries
        dicts = OrderedDict()
        for lang in sorted_langs:
            dicts[lang] = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
            if args.encoder_langtok is not None or args.decoder_langtok:
                for lang_to_add in sorted_langs:
                    dicts[lang].add_symbol(_lang_token(lang_to_add))
            print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))

        return cls(args, dicts, training)

    def _get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is None:
            return self.dicts[src_lang].eos()
        if self.args.encoder_langtok == 'src':
            return _lang_token_index(self.dicts[src_lang], src_lang)
        else:
            return _lang_token_index(self.dicts[src_lang], tgt_lang)

    def _get_decoder_langtok(self, tgt_lang):
        if not self.args.decoder_langtok:
            return self.dicts[tgt_lang].eos()
        return _lang_token_index(self.dicts[tgt_lang], tgt_lang)

    def _replace_dataset_langtok(self, lang_pair_dataset,
            src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return lang_pair_dataset

        new_src_eos = None
        if self.args.encoder_langtok is not None and src_lang is not None and tgt_lang is not None:
            new_src_eos = self._get_encoder_langtok(src_lang, tgt_lang)
        else:
            src_eos = None

        new_tgt_bos = None
        if self.args.decoder_langtok and tgt_lang is not None:
            new_tgt_bos = self._get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        if src_eos == tgt_eos:
            return TransformEosLangPairDataset(
                lang_pair_dataset,
                eos=src_eos,
                new_src_eos=new_src_eos,
                new_tgt_bos=new_tgt_bos,
            )

        dataset = lang_pair_dataset
        if src_eos is not None:
            dataset = TransformEosLangPairDataset(
                dataset,
                eos=src_eos,
                new_src_eos=new_src_eos,
            )
        if tgt_eos is not None:
            dataset = TransformEosLangPairDataset(
                dataset,
                eos=tgt_eos,
                new_tgt_bos=new_tgt_bos,
            )
        return dataset


    def load_dataset(self, split, **kwargs):
        """Load a dataset split."""

        def split_exists(split, src, tgt, lang):
            if src is not None:
                filename = os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            else:
                filename = os.path.join(self.args.data, '{}.{}-None.{}'.format(split, src, tgt))
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
                    prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
                elif split_exists(split, tgt, src, src):
                    prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
                else:
                    continue
                src_datasets[lang_pair] = indexed_dataset(prefix + src, self.dicts[src])
                tgt_datasets[lang_pair] = indexed_dataset(prefix + tgt, self.dicts[tgt])
                print('| parallel-{} {} {} examples'.format(self.args.data, split, len(src_datasets[lang_pair])))
            if len(src_datasets) == 0:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        # back translation datasets
        backtranslate_datasets = {}
        if (self.lambda_otf_bt > 0.0 or self.lambda_otf_bt_steps is not None) and split.startswith("train"):
            for lang_pair in self.args.lang_pairs:
                src, tgt = lang_pair.split('-')
                if not split_exists(split, tgt, None, tgt):
                    raise FileNotFoundError('Dataset not found: backtranslation {} ({})'.format(split, self.args.data))
                filename = os.path.join(self.args.data, '{}.{}-None.{}'.format(split, tgt, tgt))
                dataset = indexed_dataset(filename, self.dicts[tgt])
                lp_dataset = LanguagePairDataset(
                            dataset,
                            dataset.sizes,
                            self.dicts[tgt],
                            left_pad_source=self.args.left_pad_source,
                            left_pad_target=self.args.left_pad_target,
                        )
                dataset = self._replace_dataset_langtok(
                    lp_dataset,
                    src_eos=self.dicts[tgt].eos(),
                    src_lang=tgt,
                    tgt_lang=src,
                )
                backtranslate_datasets[lang_pair] = BacktranslationDataset(
                    tgt_dataset=dataset, src_dict=self.dicts[src], tgt_dict=self.dicts[tgt],
                    output_collater=self._replace_dataset_langtok(
                        lang_pair_dataset=lp_dataset,
                        src_eos=self.dicts[tgt].eos(),
                        src_lang=src,
                        tgt_eos=self.dicts[src].eos(),
                        tgt_lang=tgt,
                    ).collater,
                )
                print('| backtranslate-{}: {} {} {} examples'.format(
                    tgt, self.args.data, split, len(backtranslate_datasets[lang_pair]),
                ))
                if self.training:
                    self.backtranslate_datasets[lang_pair] = backtranslate_datasets[lang_pair]

        # denoising autoencoder
        noising_datasets = {}
        if (self.lambda_denoising > 0.0 or self.lambda_denoising_steps is not None) and split.startswith("train"):
            for lang_pair in self.args.lang_pairs:
                _, tgt = lang_pair.split('-')
                if not split_exists(split, tgt, None, tgt):
                    continue
                filename = os.path.join(self.args.data, '{}.{}-None.{}'.format(split, tgt, tgt))
                tgt_dataset1 = indexed_dataset(filename, self.dicts[tgt])
                tgt_dataset2 = indexed_dataset(filename, self.dicts[tgt])
                noising_dataset = NoisingDataset(
                    tgt_dataset1,
                    self.dicts[tgt],
                    # TODO check how to set this properly
                    seed=1,
                    max_word_shuffle_distance=self.args.max_word_shuffle_distance,
                    word_dropout_prob=self.args.word_dropout_prob,
                    word_blanking_prob=self.args.word_blanking_prob,
                )
                lang_pair_ds = LanguagePairDataset(
                    noising_dataset,
                    tgt_dataset1.sizes,
                    self.dicts[tgt],
                    tgt_dataset2,
                    tgt_dataset2.sizes,
                    self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                )
                noising_datasets[lang_pair] = self._replace_dataset_langtok(
                    lang_pair_ds,
                    src_eos=self.dicts[src].eos(),
                    src_lang=tgt,
                    tgt_eos=self.dicts[tgt].eos(),
                    tgt_lang=tgt,
                )
                print('| denoising-{}: {} {} {} examples'.format(
                    tgt, self.args.data, split, len(noising_datasets[lang_pair]),
                ))

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            src_dataset, tgt_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            return self._replace_dataset_langtok(
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

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return RoundRobinZipDatasets(
            OrderedDict([(
                lang_pair,
                self._replace_dataset_langtok(
                    LanguagePairDataset(
                        src_tokens, src_lengths,
                        self.source_dictionary
                    ),
                    src_eos=self.dicts[src].eos(),
                    src_lang=self.args.source_lang,
                    tgt_eos=self.dicts[tgt].eos(),
                    tgt_lang=self.args.target_lang,
                ),
            )]),
            eval_key=lang_pair,
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiModel):
            raise ValueError('MultilingualTranslationTask requires a FairseqMultiModel architecture')

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
                def backtranslate_fn(
                    sample, model=model.models["{}-{}".format(tgt, src)],
                    bos_token=self._get_decoder_langtok(src)):
                    return self.sequence_generators[key].generate(
                        [model],
                        sample,
                        bos_token=bos_token,
                    )
                self.backtranslate_datasets[lang_pair].set_backtranslation_fn(backtranslate_fn)

        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        # parallel data
        if self.lambda_parallel > 0.0:
            for lang_pair in self.args.lang_pairs:
                if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue

                loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
                if ignore_grad:
                    loss *= 0
                optimizer.backward(loss)
                agg_loss += loss.detach().item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output

        # online back-translation
        if self.lambda_otf_bt > 0.:
            for lang_pair in self.args.lang_pairs:
                src, tgt = lang_pair.split('-')
                sample_key = _get_bt_dataset_key(lang_pair)
                if sample_key not in sample or len(sample[sample_key]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[sample_key])
                optimizer.backward(loss)
                agg_loss += loss.detach().item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[sample_key] = logging_output
        if self.lambda_denoising > 0:
            for lang_pair in self.args.lang_pairs:
                sample_key = _get_denoising_dataset_key(lang_pair)
                if sample[sample_key] is None or len(sample[sample_key]) == 0:
                    continue
                _, tgt = lang_pair.split('-')
                loss, sample_size, logging_output = criterion(model.models['{}-{}'.format(tgt, tgt)], sample[sample_key])
                if ignore_grad:
                    loss *= 0
                else:
                    loss *= self.lambda_denoising
                optimizer.backward(loss)
                agg_loss += loss.detach().item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[sample_key] = logging_output

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang_pair in self.args.lang_pairs:
                if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=_lang_token_index(self.target_dictionary, self.args.target_lang)
                        if self.args.decoder_langtok else self.target_dictionary.eos(),
            )

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        # aggregate logging outputs for each language pair
        logging_output_keys = set(
            key
            for logging_output in logging_outputs
            for key in logging_output
        )
        lang_pair_keys = set(self.args.lang_pairs + [
            _get_bt_dataset_key(lang_pair)
            for lang_pair in self.args.lang_pairs
        ] + [
            _get_denoising_dataset_key(lang_pair)
            for lang_pair in self.args.lang_pairs
        ])
        logging_output_keys = logging_output_keys.intersection(lang_pair_keys)
        agg_logging_outputs = {
            lang_pair: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(lang_pair, {}) for logging_output in logging_outputs
            ])
            for lang_pair in logging_output_keys
        }

        def sum_over_languages(key):
            return sum(logging_output[key] if key in logging_output else 0.0 for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        if any('nll_loss' in logging_output for logging_output in agg_logging_outputs.values()):
            flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output


    def step_update(self, num_updates):
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

        self._num_updates = num_updates


    @property
    def source_dictionary(self):
        return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.args.target_lang]

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        # hack
        if len(self.datasets.values()) == 0:
            #return (1024, 1024)
            return OrderedDict([
                (lang_pair, (self.args.max_source_positions, self.args.max_target_positions))
                for lang_pair in self.args.lang_pairs
            ])
        return OrderedDict([
            (key, (self.args.max_source_positions, self.args.max_target_positions))
            for key in next(iter(self.datasets.values())).datasets.keys()
        ])
