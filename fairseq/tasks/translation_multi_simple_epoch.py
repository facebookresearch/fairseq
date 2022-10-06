# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import itertools
import json
import logging
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    ListDataset,
    data_utils,
    iterators,
)
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.multilingual_utils import LangTokStyle, get_lang_tok
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.utils import FileContentsAction


###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()


###

EVAL_BLEU_ORDER = 4
MINED_DATA_TAG = torch.tensor([45, 50, 248120, 49, 248123]).int().cpu()

logger = logging.getLogger(__name__)


@register_task("translation_multi_simple_epoch")
class TranslationMultiSimpleEpochTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')
        parser.add_argument('--one-dataset-per-batch', action='store_true',
                            help='limit each minibatch to one sub-dataset (typically lang direction)')

        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-args', default='{}',
                            help='generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string')
        parser.add_argument('--eval-bleu-all-same-batch', action='store_true',
                            help='All GPUs compute the same batch. '
                            'Required for MOE to ensure that all GPUs make the same number of all2all calls during generation')
        parser.add_argument('--eval-bleu-remove-bpe', default=None, action="store_const",
                            help="remove BPE before computing BLEU", const="@@ ")
        parser.add_argument('--eval-tokenized-bleu', action='store_true',
                            help="compute tokenized BLEU instead of sacrebleu. Use together with --eval-bleu-detok=space, to calculate spBLEU")
        parser.add_argument('--eval-bleu-detok', default='space',
                            help="detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
                            "'space' to disable detokenization, e.g., when using SPM. "
                            "use 'space' to disable detokenization; see fairseq.data.encoders for other options")
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help="print sample generations during validation")

        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        # fmt: on

    def __init__(self, args, langs, dicts, training):
        super().__init__(args)
        self.langs = langs
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
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
        self.source_langs = [d.split("-")[0] for d in self.lang_pairs]
        self.target_langs = [d.split("-")[1] for d in self.lang_pairs]
        self.check_dicts(self.dicts, self.source_langs, self.target_langs)

        self.sampling_method = SamplingMethod.build_sampler(args, self)
        self.data_manager = MultilingualDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )
        self.lang_idx = self.get_lang_idx()
        self.one_dataset_per_batch = getattr(args, "one_dataset_per_batch", False)
        self.eval_bleu = args.eval_bleu
        self.eval_bleu_args = args.eval_bleu_args
        self.eval_bleu_all_same_batch = args.eval_bleu_all_same_batch
        self.eval_bleu_remove_bpe = args.eval_bleu_remove_bpe
        self.eval_tokenized_bleu = args.eval_tokenized_bleu
        self.eval_bleu_detok = args.eval_bleu_detok
        self.eval_bleu_print_samples = args.eval_bleu_print_samples

    def get_lang_idx(self):
        lang_idx = torch.zeros(len(self.langs) + 1, dtype=torch.int32)
        # idx 0 for non-matching prefix tokens
        lang_idx[0] = -1
        for i, lang in enumerate(self.langs):
            lang_tok = get_lang_tok(lang, LangTokStyle.multilingual.value)
            lang_idx[i + 1] = MultilingualDatasetManager.get_langtok_index(
                lang_tok, self.source_dictionary
            )
        return lang_idx

    def check_dicts(self, dicts, source_langs, target_langs):
        if self.args.source_dict is not None or self.args.target_dict is not None:
            # no need to check whether the source side and target side are sharing dictionaries
            return
        src_dict = dicts[source_langs[0]]
        tgt_dict = dicts[target_langs[0]]
        for src_lang in source_langs:
            assert (
                src_dict == dicts[src_lang]
            ), "Diffrent dictionary are specified for different source languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all source languages"
        for tgt_lang in target_langs:
            assert (
                tgt_dict == dicts[tgt_lang]
            ), "Diffrent dictionary are specified for different target languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all target languages"

    @classmethod
    def setup_task(cls, args, **kwargs):
        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )
        return cls(args, langs, dicts, training)

    def has_sharded_data(self, split):
        return self.data_manager.has_sharded_data(split)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split in self.datasets:
            dataset = self.datasets[split]
            if self.has_sharded_data(split):
                if self.args.virtual_epoch_size is not None:
                    if dataset.load_next_shard:
                        shard_epoch = dataset.shard_epoch
                    else:
                        # no need to load next shard so skip loading
                        # also this avoid always loading from beginning of the data
                        return
                else:
                    shard_epoch = epoch
        else:
            # estimate the shard epoch from virtual data size and virtual epoch size
            shard_epoch = self.data_manager.estimate_global_pass_epoch(epoch)
        logger.info(f"loading data for {split} epoch={epoch}/{shard_epoch}")
        logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        if split in self.datasets:
            del self.datasets[split]
            logger.info("old dataset deleted manually")
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        split_datasets = self.data_manager.load_dataset(
            split,
            self.training,
            epoch=epoch,
            combine=combine,
            shard_epoch=shard_epoch,
            **kwargs,
        )
        for split, dataset in split_datasets.items():
            self.datasets[split] = dataset

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        src_data = ListDataset(src_tokens, src_lengths)
        dataset = LanguagePairDataset(src_data, src_lengths, self.source_dictionary)
        src_langtok_spec, tgt_langtok_spec = self.args.langtoks["main"]
        if self.args.lang_tok_replacing_bos_eos:
            dataset = self.data_manager.alter_dataset_langtok(
                dataset,
                src_eos=self.source_dictionary.eos(),
                src_lang=self.args.source_lang,
                tgt_eos=self.target_dictionary.eos(),
                tgt_lang=self.args.target_lang,
                src_langtok_spec=src_langtok_spec,
                tgt_langtok_spec=tgt_langtok_spec,
            )
        else:
            dataset.src = self.data_manager.src_dataset_transform_func(
                self.args.source_lang,
                self.args.target_lang,
                dataset=dataset.src,
                spec=src_langtok_spec,
            )
        return dataset

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        keep_langtok_in_output = getattr(args, "keep_inference_langtok", False) or (
            self.eval_bleu and models[0].training
        )
        # langtok is required for eval_bleu in order to process
        # tokenized sentence correctly
        if not keep_langtok_in_output:
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if tgt_langtok_spec:
                tgt_lang_tok = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}

        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        if self.eval_bleu:
            self.tokenizer = super().build_tokenizer(
                Namespace(tokenizer=self.eval_bleu_detok)
            )

            generation_args = json.loads(self.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model],
                Namespace(**vars(args), **generation_args),
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        # extract the langtok from the sample: assumes the langtok is prepended
        assert self.args.decoder_langtok
        prefix_tokens = sample["target"][:, 0]
        symbols_to_ignore = prefix_tokens.tolist()
        prefix_tokens = prefix_tokens.resize(prefix_tokens.size(0), 1)

        def decode(toks, escape_unk=False, remove_mined_data_tag=False):
            toks = toks.int().cpu()

            # workaround to remove <MINED_DATA> tag
            # see:
            # https://fburl.com/bzmkry8j and
            # https://github.com/fairinternal/fairseq-py/pull/3327
            if (
                remove_mined_data_tag
                and len(toks) >= 6
                and torch.equal(toks[1:6], MINED_DATA_TAG)
            ):
                toks = torch.cat((toks[0:1], toks[6:]), dim=0)

            s = self.target_dictionary.string(
                toks,
                self.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
                extra_symbols_to_ignore=symbols_to_ignore,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(
            generator, [model], sample, prefix_tokens=prefix_tokens
        )
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"], remove_mined_data_tag=True))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], force=True, tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs], force=True)

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None and tgt_langtok_spec:
                    tgt_lang_tok = self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    src_tokens = sample["net_input"]["src_tokens"]
                    bsz = src_tokens.size(0)
                    prefix_tokens = (
                        torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                    )
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
            else:
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    if tgt_langtok_spec
                    else self.target_dictionary.eos(),
                )

    def analysis_step(self, models, sample):
        model = models[0]
        model.train()
        src_tokens = sample["net_input"]["src_tokens"]
        tgt_tokens = sample["net_input"]["prev_output_tokens"]

        with torch.no_grad():
            decoder_out = model(**sample["net_input"])
            # gather metadata
            metadata, counters = gather_model_moe_metadata(
                model,
                len(self.source_dictionary),
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
            )
        return metadata, counters

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        return self.data_manager.get_source_dictionary(self.source_langs[0])

    @property
    def target_dictionary(self):
        return self.data_manager.get_target_dictionary(self.target_langs[0])

    def split_for_dataset(self, dataset):
        splits = [s for s, _ in self.datasets.items() if self.datasets[s] == dataset]
        return splits[0] if len(splits) > 0 else None

    def create_batch_sampler_func(
        self,
        max_positions,
        ignore_invalid_inputs,
        max_tokens,
        max_sentences,
        required_batch_size_multiple=1,
        seed=1,
    ):
        def construct_batch_sampler(dataset, epoch):
            split = self.split_for_dataset(dataset)
            # NEW implementation
            if epoch is not None:
                # initialize the dataset with the correct starting epoch
                dataset.set_epoch(epoch)

            # get indices ordered by example size
            start_time = time.time()
            logger.info(f"start batch sampler: mem usage: {data_utils.get_mem_usage()}")

            with data_utils.numpy_seed(seed):
                if self.one_dataset_per_batch:
                    ordered_indices_list = dataset.ordered_indices_per_dataset()
                else:
                    ordered_indices_list = [dataset.ordered_indices()]

            # get batches constructed from each underlying dataset to concatenate
            subdataset_sampler_list = []
            for ds_idx, indices in enumerate(ordered_indices_list):
                if self.one_dataset_per_batch:
                    log_tag = f"[{split}] [{ds_idx}]"
                else:
                    log_tag = f"[{split}]"
                logger.info(
                    f"{log_tag} @batch_sampler order indices time: {get_time_gap(start_time, time.time())}"
                )
                logger.info(f"mem usage: {data_utils.get_mem_usage()}")

                # filter examples that are too large
                if max_positions is not None and split is not None:
                    my_time = time.time()
                    indices = self.filter_indices_by_size(
                        indices, dataset, max_positions, ignore_invalid_inputs
                    )
                    logger.info(
                        f"{log_tag} @batch_sampler filter_by_size time: {get_time_gap(my_time, time.time())}"
                    )
                    logger.info(f"mem usage: {data_utils.get_mem_usage()}")

                # create mini-batches with given size constraints
                my_time = time.time()
                batch_sampler = dataset.batch_by_size(
                    indices,
                    max_tokens=max_tokens,
                    max_sentences=max_sentences,
                    required_batch_size_multiple=required_batch_size_multiple,
                )
                subdataset_sampler_list.append(batch_sampler)

                end_time = time.time()
                logger.info(
                    f"{log_tag} @batch_sampler batch_by_size time: {get_time_gap(my_time, end_time)}"
                )
                logger.info(
                    f"{log_tag} per epoch batch_sampler set-up time: {get_time_gap(start_time, end_time)}"
                )
                logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            combined_batch_sampler = itertools.chain(*subdataset_sampler_list)
            return combined_batch_sampler

        return construct_batch_sampler

    # we need to override get_batch_iterator because we want to reset the epoch iterator each time
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # initialize the dataset with the correct starting epoch
        assert isinstance(dataset, FairseqDataset)
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]

        # required for MOE
        split = self.split_for_dataset(dataset)
        is_valid = "valid" in split if split is not None else False
        if self.eval_bleu and self.eval_bleu_all_same_batch and is_valid:
            num_shards = 1
            shard_id = 0

        if self.args.sampling_method == "RoundRobin":
            batch_iter = super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
                skip_remainder_batch=skip_remainder_batch,
                update_epoch_batch_itr=update_epoch_batch_itr,
            )
            self.dataset_to_epoch_iter[dataset] = batch_iter
            return batch_iter

        construct_batch_sampler = self.create_batch_sampler_func(
            max_positions,
            ignore_invalid_inputs,
            max_tokens,
            max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
        )

        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=construct_batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            skip_remainder_batch=skip_remainder_batch,
        )
        return epoch_iter


def gather_model_moe_metadata(model, vocab_size, src_tokens=None, tgt_tokens=None):
    from fairseq.modules.moe import MOELayer

    def shorten_name(name):
        for sub in ["_fsdp_wrapped_module.", "_fpw_module.", "layers.", ".moe_layer"]:
            name = name.replace(sub, "")
        return name

    moe_logging_output = {}
    counters = {}

    for key in ["soft_gates"]:
        for name, module in model.named_modules():
            if isinstance(module, MOELayer):
                if key in module.metadata:
                    # assuming one-dataset per batch:
                    val = module.metadata[key]
                    num_tokens = val.shape[0]
                    val = val.sum(dim=0)
                    moe_logging_output[f"{key}_{shorten_name(name)}"] = val
                    counters[f"{key}_{shorten_name(name)}"] = num_tokens

    # Experimental: looking at average expert assignment for each token in the vocabulary
    # It is slow and dumps large vectors (V * E)
    if 0:
        # stats for each vocab token:
        # sample src hot one
        src_tokens = F.one_hot(src_tokens.reshape(-1), vocab_size)
        # sample tgt hot one
        tgt_tokens = F.one_hot(tgt_tokens.reshape(-1), vocab_size)

        key = "soft_gates"
        for name, module in model.named_modules():
            if isinstance(module, MOELayer):
                if key in module.metadata:
                    # assuming one-dataset per batch:
                    val = module.metadata[key]

                    # aggregate by token index before summing
                    if "encoder" in name:
                        token_gates = torch.mm(src_tokens.t().float(), val)
                        num_tokens_by_index = src_tokens.sum(dim=0)
                        print("num tok per index", num_tokens_by_index.shape)
                    else:
                        token_gates = torch.mm(tgt_tokens.t().float(), val)  # V,E
                        num_tokens_by_index = tgt_tokens.sum(dim=0)  # V
                        print("num tok per index", num_tokens_by_index.shape)

                    moe_logging_output[
                        f"token_gates_{shorten_name(name)}"
                    ] = token_gates
                    counters[f"token_gates_{shorten_name(name)}"] = num_tokens_by_index

    return moe_logging_output, counters
