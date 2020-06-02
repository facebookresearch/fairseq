# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch

from fairseq.data import LanguagePairDataset

from .translation import load_langpair_dataset, TranslationTask
from . import register_task
from .. import utils

logger = logging.getLogger(__name__)

@register_task('translation_from_pretrained_bart')
class TranslationFromPretrainedBARTTask(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

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
        TranslationTask.add_args(parser)
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, for example, "en,de,fr"'
                                 'be careful these langs are what you used for pretraining (the same order),'
                                 'not for finetuning.'
                                 'you should always add all pretraining language idx during finetuning.')

        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence. '
                                 'this aligns with how mBART was pretrained.')

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(',')
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol('[{}]'.format(l))
            d.add_symbol('<mask>')

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, 'max_source_positions', 1024),
            max_target_positions=getattr(self.args, 'max_target_positions', 1024),
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, 'prepend_bos', False),
            append_source_id=True
            )

    def build_generator(self, models, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index('[{}]'.format(self.args.target_lang))
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                eos=self.tgt_dict.index('[{}]'.format(self.args.target_lang))
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_lang_id = self.source_dictionary.index('[{}]'.format(self.args.source_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)
        return dataset

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False, **kwargs):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ), **kwargs
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        eos = self.tgt_dict.index('[{}]'.format(self.args.target_lang))
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens'],
                               extra_symbols_to_ignore=[eos]))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
                extra_symbols_to_ignore=[eos]
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

