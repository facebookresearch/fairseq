# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import logging
import math
import os
from argparse import Namespace
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fairseq
from fairseq import metrics, options, utils
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    NoisingDataset,
    PrependTokenDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
    data_utils,
    encoders,
)
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

logger = logging.getLogger(__name__)


class PiecewiseLinearFn:
    """Piecewise linear function. Can be configured with a string."""

    def __init__(self, pieces: Sequence[Tuple[int, float]]):
        assert pieces == sorted(
            pieces
        ), f"PiecewiseLinearFn configuration should be sorted, received: {pieces}"

        self.pieces = pieces

    def __call__(self, x: int) -> float:
        for i, (x_a, y_a) in enumerate(self.pieces[:-1]):
            x_b, y_b = self.pieces[i + 1]
            if x_a <= x <= x_b:
                return y_a + (x - x_a) * (y_b - y_a) / (x_b - x_a)

        return self.pieces[-1][1]

    @staticmethod
    def from_string(configuration: str) -> "PiecewiseLinearFn":
        """
        Parse the configuration of lambda coefficient (for scheduling).
        x = "3"                  # lambda will be a constant equal to x
        x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                                 # to 0 during the first 1000 iterations
        x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                                 # iterations, then will linearly increase to 1 until iteration 2000
        """
        if isinstance(configuration, float):
            return PiecewiseLinearFn([(0, configuration)])

        try:
            parts = configuration.split(",")
            if len(parts) == 1:
                v = float(configuration)
                return PiecewiseLinearFn([(0, v)])

            split = [s.split(":") for s in parts]
            pieces = [(int(t), float(v)) for t, v in split]
            return PiecewiseLinearFn(pieces)
        except Exception:
            raise ValueError(
                f"Invalid PiecewiseLinearFn configuration: {configuration!r}"
            )

    @staticmethod
    def one() -> "PiecewiseLinearFn":
        return PiecewiseLinearFn([(0, 1.0)])


@register_task("online_backtranslation")
class OnlineBackTranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # Generic translation args
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('--mono-langs', metavar='MONO_LANGS',
                            help='monolingual languages for training')
        parser.add_argument('--valid-lang-pairs', default=None, metavar='VALID_LANG_PAIRS',
                            help='language pairs for validation')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # Denoising args
        parser.add_argument('--max-word-shuffle-distance', default=3.0, type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.2, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')

        # Backtranslation args
        parser.add_argument('--lambda-bt', default="1.0", type=str, metavar='N',
                            help='back-translation weight')
        parser.add_argument('--lambda-dae', default="1.0", type=str, metavar='N',
                            help='denoising auto-encoder weight')

        # Evaluation args
        parser.add_argument('--generate-one-by-one', action='store_true',
                            help='generate one sentence at a time for backtranslation')

        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on

    def __init__(self, args, common_dict, mono_langs, valid_lang_pairs):
        super().__init__(args, common_dict, common_dict)
        self.common_dict = common_dict
        self.mono_langs = mono_langs
        self.valid_lang_pairs = valid_lang_pairs

        self.SHOW_SAMPLES_INTERVAL = 1000
        # Start by showing samples
        self._show_samples_ctr = self.SHOW_SAMPLES_INTERVAL
        self.SHOW_SAMPLES_NUMBER = 5
        self.lambda_bt = PiecewiseLinearFn.from_string(args.lambda_bt)
        self.lambda_dae = PiecewiseLinearFn.from_string(args.lambda_dae)

        self.args = args
        self.data = utils.split_paths(self.args.data)
        if len(self.data) == 1:
            shards = list(Path(self.data[0]).glob("shard*"))
            if len(shards) > 0:
                # keep this as strings, since it can also be a manifold path
                old_data = self.data
                self.data = [str(shard) for shard in shards]
                logging.warning(f"Expanded data directory {old_data} to {self.data}")

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        assert args.mono_langs is not None

        mono_langs = args.mono_langs.split(",")
        valid_lang_pairs = args.valid_lang_pairs.split(",")

        # load dictionary
        dict_path = os.path.join(paths[0], "dict.txt")
        common_dict = cls.load_dictionary(dict_path)

        return cls(args, common_dict, mono_langs, valid_lang_pairs)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs) -> FairseqDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split == "train":
            data_path = self.data[(epoch - 1) % len(self.data)]
            dataset = self.load_train_dataset(data_path)
        else:
            # valid/test should always be the same.
            dataset = self.load_translation_dataset(split, self.data[0])

        self.datasets[split] = dataset
        return dataset

    def load_train_dataset(self, data_path: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset and denoising dataset."""
        data = []
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            # TODO: could we do the BT using denoise sample ?
            # this would half the data loading work
            data.append((f"{lang}-BT", self.load_bt_dataset(train_path, lang)))
            data.append(
                (f"{lang}-DENOISE", self.load_denoise_dataset(train_path, lang))
            )

        return RoundRobinZipDatasets(OrderedDict(data))

    def _langpair_dataset(
        self, src: FairseqDataset, tgt: FairseqDataset
    ) -> LanguagePairDataset:
        return LanguagePairDataset(
            src,
            src.sizes,
            self.dictionary,
            tgt=tgt,
            tgt_sizes=tgt.sizes,
            tgt_dict=self.dictionary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            # TODO: should we shuffle ? we are already sorting batch by sizes so ?
            # shuffle=True,
        )

    def _prepend_lang_bos_to_target(
        self, dataset: LanguagePairDataset, lang: str
    ) -> LanguagePairDataset:
        bos = _lang_token_index(self.dictionary, lang)
        return TransformEosLangPairDataset(
            dataset,
            src_eos=self.dictionary.eos(),
            new_src_eos=self.dictionary.eos(),
            tgt_bos=self.dictionary.eos(),
            new_tgt_bos=bos,
        )

    def load_bt_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """The BT dataset is generated with (tgt, tgt) pairs.
        The actual translation to a (generated_src, tgt) pair
        is done on the fly during training.
        """
        mono_dataset = data_utils.load_indexed_dataset(
            data_path, self.common_dict, self.args.dataset_impl
        )
        assert mono_dataset is not None, f"No dataset found for {lang}"

        mono_dataset_src = PrependTokenDataset(
            mono_dataset, _lang_token_index(self.dictionary, lang)
        )

        mono_dataset_bt = self._langpair_dataset(mono_dataset_src, mono_dataset)
        logger.info(
            f"mono_lang = {lang} "
            f"lang token index = {_lang_token_index(self.dictionary, lang)} "
            f"lang token = {_lang_token(lang)}"
        )

        mono_dataset_bt = self._prepend_lang_bos_to_target(mono_dataset_bt, lang)
        return mono_dataset_bt

    def load_denoise_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """Classic denoising dataset"""
        dataset = data_utils.load_indexed_dataset(
            data_path, self.common_dict, self.args.dataset_impl
        )
        noisy_dataset = NoisingDataset(
            dataset,
            self.dictionary,
            seed=1,
            max_word_shuffle_distance=self.args.max_word_shuffle_distance,
            word_dropout_prob=self.args.word_dropout_prob,
            word_blanking_prob=self.args.word_blanking_prob,
        )
        noisy_dataset = PrependTokenDataset(
            noisy_dataset, _lang_token_index(self.dictionary, lang)
        )

        clean_dataset = data_utils.load_indexed_dataset(
            data_path, self.common_dict, self.args.dataset_impl
        )
        denoising_dataset = self._langpair_dataset(noisy_dataset, clean_dataset)
        denoising_dataset = self._prepend_lang_bos_to_target(denoising_dataset, lang)
        return denoising_dataset

    def load_translation_dataset(
        self, split: str, data_path: str, combine: bool = False
    ):
        # only judging with one language pair for the moment,
        # since ConcatDataset doesn't work as expected
        assert len(self.valid_lang_pairs) == 1, "For now..."
        valid_lang_pair = self.valid_lang_pairs[0]
        src, tgt = valid_lang_pair.split("-")

        # use the same function than TranslationTask
        src_tgt_dt = load_langpair_dataset(
            data_path,
            split,
            src,
            self.common_dict,
            tgt,
            self.common_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            prepend_bos_src=_lang_token_index(self.dictionary, src),
        )

        src_tgt_eos_dt = self._prepend_lang_bos_to_target(src_tgt_dt, tgt)
        src_tgt_eos_dt.args = self.args
        return src_tgt_eos_dt

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        raise NotImplementedError

    def build_model(self, args):
        # torch.autograd.set_detect_anomaly(True)
        model = super().build_model(args)

        add_secial_tokens_to_dict_and_model(self.common_dict, model, self.mono_langs)

        self.sequence_generators = {}
        for mono_lang in self.mono_langs:
            self.sequence_generators[mono_lang] = SequenceGenerator(
                [model],
                tgt_dict=self.dictionary,
                beam_size=1,
                max_len_a=1.3,
                max_len_b=5,
                min_len=5,
                # keep 1 to be able to prepend bos
                max_len=model.max_decoder_positions() - 1,
            )

        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.bleu_sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.common_dict

    def display_samples_once_in_a_while(self, smp, mono_lang, other_lang):
        self._show_samples_ctr += 1
        if self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
            return
        self._show_samples_ctr = 0

        ln = smp["net_input"]["src_tokens"].shape[0]

        logger.info(
            f"(r:{self.args.distributed_rank}) : "
            f"{other_lang} ---> {mono_lang} "
            f"({other_lang} was generated by back-translation.) {ln} samples"
        )

        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]

            src_str = self.dictionary.string(src_tokens, "sentencepiece")
            tgt_str = self.dictionary.string(tgt_tokens, "sentencepiece")
            logger.info(
                f"\n{i}\t\t[{other_lang} generated]  {src_str}\n"
                f"\t\t[{mono_lang} original ]  {tgt_str}\n"
                f"\t\t[ src tokens]  {src_tokens}\n"
            )

    def backtranslate_sample(self, smp, orig_lang, other_lang) -> None:
        """
        * WARNING: smp is modified in place.
        * At the start of this function, `smp` has the same input and target:
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (from data) __en__ hello world |  __en__ hello world   |
          |--------------------------------------------------------|

        * We call generator.generate(smp, bos_token = token("ro")),
        and copy the result as input
        * At the end, `smp` has the translation to other language.
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (generated) __ro__ salut lume  |  __en__ hello world   |
          |--------------------------------------------------------|

        """
        bos_token = _lang_token_index(self.dictionary, other_lang)
        generated = self.sequence_generators[orig_lang].generate(
            models=[], sample=smp, bos_token=bos_token
        )

        max_lngth = max([gn[0]["tokens"].size(0) for gn in generated])
        net_input = smp["net_input"]
        n_src_tokens = torch.empty(
            size=(len(generated), max_lngth + 1), dtype=net_input["src_tokens"].dtype
        )
        n_src_lengths = torch.empty(
            len(generated), dtype=net_input["src_lengths"].dtype
        )

        for i, gn in enumerate(generated):
            tokens = gn[0]["tokens"]
            tokens_size = tokens.size(0)
            padding_needed = max_lngth - tokens_size
            tokens = torch.cat([tokens.new([bos_token]), tokens])
            tokens = F.pad(tokens, (0, padding_needed), value=self.dictionary.pad())
            n_src_tokens[i] = tokens
            n_src_lengths[i] = tokens_size + 1

        device = net_input["src_tokens"].device
        # This seems to be important
        del net_input["src_tokens"]
        del net_input["src_lengths"]
        net_input["src_tokens"] = n_src_tokens.to(device)
        net_input["src_lengths"] = n_src_lengths.to(device)

    def generate(self, smp, model):
        model.eval()
        orig_lang = (
            self.dictionary[smp["net_input"]["src_tokens"][0][0]]
            .replace(" ", "")
            .replace("_", "")
        )
        bos_token = smp["net_input"]["prev_output_tokens"][0][0]
        with torch.no_grad():
            generated = self.sequence_generators[orig_lang].generate(
                models=[model], sample=smp, bos_token=bos_token
            )
        return generated

    def get_other_lang(self, lang):
        # TODO: allow more complex mapping
        if lang != self.mono_langs[0]:
            return self.mono_langs[0]
        if len(self.mono_langs) == 2:
            return self.mono_langs[1]
        return self.mono_langs[np.random.randint(1, len(self.mono_langs))]

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        dataset_keys = self.datasets["train"].datasets.keys()

        weights = {
            "BT": self.lambda_bt(update_num),
            "DENOISE": self.lambda_dae(update_num),
        }
        log_keys = {"BT": "bt_", "DENOISE": "dae_"}

        for dataset_key in dataset_keys:
            smp = sample[dataset_key]
            mono_lang, task_subtype = dataset_key.split("-")
            if weights[task_subtype] == 0:
                continue

            if task_subtype == "BT":
                with torch.autograd.profiler.record_function("backtranslation"):
                    model.eval()
                    # TODO: Could we translate to several language at once ?
                    # this would allow to share encoder_out and maximize GPU usage.
                    other_lang = self.get_other_lang(mono_lang)
                    self.backtranslate_sample(smp, mono_lang, other_lang)
                    self.display_samples_once_in_a_while(smp, mono_lang, other_lang)
                    model.train()

            # Like in FairseqTask.train_step
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, smp)
            loss *= weights[task_subtype]
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)

            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[log_keys[task_subtype] + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output

    def get_bos_token_from_sample(self, sample):
        net_input = sample["net_input"]
        source_lang_token_id = torch.unique(net_input["src_tokens"][:, 0]).item()
        source_lang_token = self.dictionary[source_lang_token_id].replace("_", "")
        target_lang_token_id = _lang_token_index(
            self.dictionary, self.get_other_lang(source_lang_token)
        )

        return target_lang_token_id

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        bt_sample_size = sum(x.get("bt_sample_size", 0) for x in logging_outputs)
        if bt_sample_size:
            bt_loss_sum = sum(x.get("bt_loss", 0) for x in logging_outputs)
            bt_loss_sum *= 1 / bt_sample_size / math.log(2)
            metrics.log_scalar("bt_loss", bt_loss_sum, bt_sample_size, round=3)

            bt_nll_loss_sum = sum(x.get("bt_nll_loss", 0) for x in logging_outputs)
            bt_ntokens = sum(x.get("bt_ntokens", 0) for x in logging_outputs)
            bt_nll_loss_sum *= 1 / bt_ntokens / math.log(2)
            metrics.log_scalar("bt_nll_loss", bt_nll_loss_sum, bt_ntokens, round=3)
            metrics.log_derived(
                "bt_ppl", lambda meters: utils.get_perplexity(meters["bt_nll_loss"].avg)
            )

        dae_sample_size = sum(x.get("dae_sample_size", 0) for x in logging_outputs)
        if dae_sample_size:
            dae_loss_sum = sum(x.get("dae_loss", 0) for x in logging_outputs)
            dae_loss_sum *= 1 / dae_sample_size / math.log(2)
            metrics.log_scalar("dae_loss", dae_loss_sum, dae_sample_size, round=3)

            dae_nll_loss_sum = sum(x.get("dae_nll_loss", 0) for x in logging_outputs)
            dae_ntokens = sum(x.get("dae_ntokens", 0) for x in logging_outputs)
            dae_nll_loss_sum *= 1 / dae_ntokens / math.log(2)
            metrics.log_scalar("dae_nll_loss", dae_nll_loss_sum, dae_ntokens, round=3)
            metrics.log_derived(
                "dae_ppl",
                lambda meters: utils.get_perplexity(meters["dae_nll_loss"].avg),
            )


@torch.no_grad()
def extend_embedding(
    emb: nn.Module, new_vocab_size: int, copy_from_token_id: int
) -> None:
    old_emb_data = emb.weight.data
    (old_vocab_size, dim) = old_emb_data.shape
    assert new_vocab_size >= old_vocab_size

    if new_vocab_size > old_vocab_size:
        emb.weight.data = torch.zeros((new_vocab_size, dim))
        emb.weight.data[:old_vocab_size, :] = old_emb_data
        # initialize new embeddings
        emb.weight.data[old_vocab_size:, :] = old_emb_data[copy_from_token_id]
        if hasattr(emb, "num_embeddings"):
            emb.num_embeddings = new_vocab_size
        if hasattr(emb, "out_features"):
            emb.out_features = new_vocab_size

    if getattr(emb, "bias", None) is None:
        return

    # Fix the bias.
    # Bias shape can be different from the previous vocab size
    # if the weight matrix was shared and alread extended but not the bias.
    (old_vocab_size,) = emb.bias.shape
    assert new_vocab_size >= old_vocab_size
    if new_vocab_size > old_vocab_size:
        old_bias = emb.bias.data
        new_bias = torch.zeros(
            (new_vocab_size,), dtype=old_bias.dtype, device=old_bias.device
        )
        new_bias[:old_vocab_size] = old_bias
        emb.bias.data = new_bias


def add_secial_tokens_to_dict_and_model(
    dictionary: "fairseq.data.Dictionary",
    model: nn.Module,
    mono_langs: Sequence[str],
) -> None:
    embs = model.encoder.embed_tokens
    vocab_size, embedding_dim = embs.weight.shape

    # The model may or may not have a '<mask>' embedding yet
    assert (
        len(dictionary) <= vocab_size <= len(dictionary) + 1
    ), f"Dictionary len ({len(dictionary)}) doesn't match embs shape ({embs.weight.shape})"
    # TODO: we should reuse the pretrained model dict which already has <mask>
    dictionary.add_symbol("<mask>")

    for lang in mono_langs:
        lang_token = _lang_token(lang)
        dictionary.add_symbol(lang_token)
    logger.info(
        f"dictionary: {len(dictionary)} -> {vocab_size} tokens "
        f"after adding {len(mono_langs)} lang tokens."
    )

    if len(dictionary) <= vocab_size:
        return

    extend_embedding(embs, len(dictionary), dictionary.bos())
    dec_embs = model.decoder.embed_tokens
    extend_embedding(dec_embs, len(dictionary), dictionary.bos())
    lm_head = model.decoder.output_projection
    extend_embedding(lm_head, len(dictionary), dictionary.bos())
    assert lm_head.weight.shape == (len(dictionary), embedding_dim)


def _lang_token(lang: str) -> str:
    return f"__{lang}__"


def _lang_token_index(dictionary, lang: str) -> int:
    return dictionary.index(_lang_token(lang))


@contextlib.contextmanager
def assert_weights_have_changed(model: nn.Module):
    def checksum(model: nn.Module) -> float:
        return sum(p.sum().item() for p in model.parameters())

    initial_checksum = checksum(model)
    yield model
    final_checksum = checksum(model)
    logger.info(
        f"initial_checksum={initial_checksum} -> final_checksum={final_checksum}"
    )
    assert initial_checksum != final_checksum, "Model hasn't changed !"
