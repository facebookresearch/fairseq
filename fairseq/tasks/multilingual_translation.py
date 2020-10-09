# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import logging
import os
from fairseq import options
import contextlib
import torch

from fairseq import metrics, utils
from fairseq.data import (
    Dictionary,
    LanguagePairDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
)
from fairseq.models import FairseqMultiModel
from fairseq.tasks.translation import load_langpair_dataset
from examples.latent_depth.src.loss.latent_depth import LatentLayersKLLoss, LatentLayersSparsityLoss 

from . import register_task, LegacyFairseqTask

logger = logging.getLogger(__name__)


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


@register_task('multilingual_translation')
class MultilingualTranslationTask(LegacyFairseqTask):
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
    `--target-lang`, which indicates the inference langauge direction.
    `--lang-pairs`, `--encoder-langtok`, `--decoder-langtok` have to be set to
    the same value as training.
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
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt)')
        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')
        parser.add_argument('--encoder-latent-layer', action='store_true', help='latent layer selection in encoder')
        parser.add_argument('--decoder-latent-layer', action='store_true', help='latent layer selection in decoder')
        parser.add_argument('--target-layers', default=-1, type=int,
                            help='number of effective layers to learn; -1 means no constraint')
        parser.add_argument('--sparsity-weight', default=0.0, type=float,
                            help='weight for sparsity loss')
        parser.add_argument('--share-weight', default=0.0, type=float,
                            help='weight for sharing loss')
        parser.add_argument('--soft-update', default=1, type=int,
                            help='number of updates with soft sampling')
        parser.add_argument('--anneal-updates', default=1, type=int,
                            help='number of updates to anneal the KL loss weight')
        parser.add_argument('--prior', default="uniform", type=str,
                            help='prior used for computing KL loss')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        src_langs, tgt_langs = zip(*[(lang.split("-")[0], lang.split("-")[1]) for lang in args.lang_pairs])
        self.src_lang_idx_dict = {lang: lang_idx for lang_idx, lang in enumerate(src_langs)}
        self.tgt_lang_idx_dict = {lang: lang_idx for lang_idx, lang in enumerate(tgt_langs)}
        self.encoder_latent_layer = hasattr(self.args, "encoder_latent_layer") and self.args.encoder_latent_layer
        if self.encoder_latent_layer:
            assert self.args.share_encoders
        self.decoder_latent_layer = hasattr(self.args, "decoder_latent_layer") and self.args.decoder_latent_layer
        if self.decoder_latent_layer:
            assert self.args.share_decoders
        self.training = training
        if training or self.encoder_latent_layer or self.decoder_latent_layer:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        if self.training and (self.encoder_latent_layer or self.decoder_latent_layer):
            self.kl_loss = LatentLayersKLLoss(self.args)
            self.sparsity_loss = LatentLayersSparsityLoss(self.args)
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.langs = list(dicts.keys())

    @classmethod
    def setup_task(cls, args, **kwargs):
        dicts, training = cls.prepare(args, **kwargs)
        return cls(args, dicts, training)

    @classmethod
    def prepare(cls, args, **kargs):
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        if args.lang_pairs is None:
            raise ValueError('--lang-pairs is required. List all the language pairs in the training objective.')
        if isinstance(args.lang_pairs, str):
            args.lang_pairs = args.lang_pairs.split(',')
        sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True

        # load dictionaries
        dicts = OrderedDict()
        for lang in sorted_langs:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dicts[lang] = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
            if args.encoder_langtok is not None or args.decoder_langtok:
                for lang_to_add in sorted_langs:
                    dicts[lang].add_symbol(_lang_token(lang_to_add))
            logger.info('[{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        return dicts, training

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is None:
            return self.dicts[src_lang].eos()
        if self.args.encoder_langtok == 'src':
            return _lang_token_index(self.dicts[src_lang], src_lang)
        else:
            return _lang_token_index(self.dicts[src_lang], tgt_lang)

    def get_decoder_langtok(self, tgt_lang):
        if not self.args.decoder_langtok:
            return self.dicts[tgt_lang].eos()
        return _lang_token_index(self.dicts[tgt_lang], tgt_lang)

    def alter_dataset_langtok(self, lang_pair_dataset,
                              src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return lang_pair_dataset

        new_src_eos = None
        if self.args.encoder_langtok is not None and src_eos is not None \
           and src_lang is not None and tgt_lang is not None:
            new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang)
        else:
            src_eos = None

        new_tgt_bos = None
        if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
            new_tgt_bos = self.get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a dataset split."""
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            langpair_dataset = load_langpair_dataset(
                data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
                combine=True, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            return self.alter_dataset_langtok(
                langpair_dataset,
                src_eos=self.dicts[src].eos(),
                src_lang=src,
                tgt_eos=self.dicts[tgt].eos(),
                tgt_lang=tgt,
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.lang_pairs
            ]),
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError("Constrained decoding with the multilingual_translation task is not supported")

        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return RoundRobinZipDatasets(
            OrderedDict([(
                lang_pair,
                self.alter_dataset_langtok(
                    LanguagePairDataset(
                        src_tokens, src_lengths,
                        self.source_dictionary
                    ),
                    src_eos=self.source_dictionary.eos(),
                    src_lang=self.args.source_lang,
                    tgt_eos=self.target_dictionary.eos(),
                    tgt_lang=self.args.target_lang,
                ),
            )]),
            eval_key=lang_pair,
        )

    def build_model(self, args):
        def check_args():
            messages = []
            if len(set(self.args.lang_pairs).symmetric_difference(args.lang_pairs)) != 0:
                messages.append('--lang-pairs should include all the language pairs {}.'.format(args.lang_pairs))
            if self.args.encoder_langtok != args.encoder_langtok:
                messages.append('--encoder-langtok should be {}.'.format(args.encoder_langtok))
            if self.args.decoder_langtok != args.decoder_langtok:
                messages.append('--decoder-langtok should {} be set.'.format("" if args.decoder_langtok else "not"))

            if len(messages) > 0:
                raise ValueError(' '.join(messages))

        # Check if task args are consistant with model args
        check_args()

        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiModel):
            raise ValueError('MultilingualTranslationTask requires a FairseqMultiModel architecture')
        return model

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        from collections import defaultdict
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)
        curr_lang_pairs = [
            lang_pair
            for lang_pair in self.model_lang_pairs
            if sample[lang_pair] is not None and len(sample[lang_pair]) != 0
        ]

        for idx, lang_pair in enumerate(curr_lang_pairs):
            src, tgt = lang_pair.split("-")
            if self.encoder_latent_layer:
                src_lang_idx = self.src_lang_idx_dict[src]
                model.models[lang_pair].encoder.set_lang_idx(src_lang_idx)
                model.models[lang_pair].encoder.layer_select.hard_select = update_num > self.args.soft_update
            if self.decoder_latent_layer:
                tgt_lang_idx = self.tgt_lang_idx_dict[tgt]
                model.models[lang_pair].decoder.set_lang_idx(tgt_lang_idx)
                model.models[lang_pair].decoder.layer_select.hard_select = update_num > self.args.soft_update

            def maybe_no_sync():
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(model, 'no_sync')
                    and idx < len(curr_lang_pairs) - 1
                ):
                    return model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager
            with maybe_no_sync():
                loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
                if self.encoder_latent_layer:
                    none_samples = sum(
                        1 if x is None else 0 for x in model.models[lang_pair].encoder.layer_select.layer_samples
                    )
                    if none_samples == 0 or self.args.prior != "agged_posterior":
                        loss += self.kl_loss(
                            model.models[lang_pair].encoder.layer_select.layer_samples,
                            src_lang_idx,
                            update_num,
                            sample_size
                        )
                if self.decoder_latent_layer:
                    none_samples = sum(
                        1 if x is None else 0 for x in model.models[lang_pair].decoder.layer_select.layer_samples
                    )
                    if none_samples == 0 or self.args.prior != "agged_posterior":
                        loss += self.kl_loss(
                            model.models[lang_pair].decoder.layer_select.layer_samples,
                            tgt_lang_idx,
                            update_num,
                            sample_size
                        )
                if ignore_grad:
                    loss *= 0

                if hasattr(self, sparsity_loss) and self.sparsity_loss.is_valid(update_num):
                    # need to retain the graph if sparsity loss needs to be added
                    loss.backward(retain_graph=True)
                else:
                    optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{lang_pair}:{k}"] += logging_output[k]

        # compute auxiliary loss from layere sparsity, based on all samples from all languages
        if hasattr(self, sparsity_loss) and self.sparsity_loss.is_valid(update_num):
            sparsity_loss = 0
            if self.encoder_latent_layer:
                sparsity_loss += self.sparsity_loss(model.models[lang_pair].encoder.layer_select.layer_samples, update_num, agg_sample_size)
            if self.decoder_latent_layer:
                sparsity_loss += self.sparsity_loss(model.models[lang_pair].decoder.layer_select.layer_samples, update_num, agg_sample_size)
            if sparsity_loss > 0:
                optimizer.backward(sparsity_loss)
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            from collections import defaultdict
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)
            for lang_pair in self.eval_lang_pairs:
                if lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                for k in logging_output:
                    agg_logging_output[k] += logging_output[k]
                    agg_logging_output[f"{lang_pair}:{k}"] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        with torch.no_grad():
            if self.encoder_latent_layer or self.decoder_latent_layer:
                for model in models:
                    if self.encoder_latent_layer:
                        assert model.encoder.layer_select is not None
                        src_lang_idx = self.src_lang_idx_dict[self.args.source_lang]
                        model.encoder.set_lang_idx(src_lang_idx)
                    if self.decoder_latent_layer:
                        assert model.decoder.layer_select is not None
                        tgt_lang_idx = self.tgt_lang_idx_dict[self.args.target_lang]
                        model.decoder.set_lang_idx(tgt_lang_idx)
            if self.args.decoder_langtok:
                bos_token = _lang_token_index(self.target_dictionary, self.args.target_lang)
            else:
                bos_token = self.target_dictionary.eos()
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=bos_token,
            )

    def reduce_metrics(self, logging_outputs, criterion):
        with metrics.aggregate():
            # pass 'sample_size', 'nsentences', 'ntokens' stats to fairseq_task
            super().reduce_metrics(logging_outputs, criterion)
            for k in ['sample_size', 'nsentences', 'ntokens']:
                metrics.log_scalar(k, sum(l[k] for l in logging_outputs))

    @property
    def source_dictionary(self):
        if self.training:
            return next(iter(self.dicts.values()))
        else:
            return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        if self.training:
            return next(iter(self.dicts.values()))
        else:
            return self.dicts[self.args.target_lang]

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if len(self.datasets.values()) == 0:
            return {'%s-%s' % (self.args.source_lang, self.args.target_lang):
                    (self.args.max_source_positions, self.args.max_target_positions)}
        return OrderedDict([
            (key, (self.args.max_source_positions, self.args.max_target_positions))
            for split in self.datasets.keys()
            for key in self.datasets[split].datasets.keys()
        ])
