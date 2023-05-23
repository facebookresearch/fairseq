# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset, TransformEosLangPairDataset

from . import register_task
from .translation import TranslationTask, load_langpair_dataset
from fairseq.data.multilingual.multilingual_utils import (
    EncoderLangtok,
    LangTokSpec,
    LangTokStyle,
    augment_dictionary,
    get_lang_tok,
)
from fairseq_cli.generate import get_symbols_to_strip_from_output
import logging

logger = logging.getLogger(__name__)

@register_task("translation_from_pretrained_multi_bart")
class TranslationFromPretrainedMultiBARTTask(TranslationTask):
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
        parser.add_argument('--langs',  type=str, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')

                        
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.args = args
        self.langs = args.langs.split(",")
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol("[{}]".format(l))
            d.add_symbol("<mask>")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        ds = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, "max_source_positions", 1024),
            max_target_positions=getattr(self.args, "max_target_positions", 1024),
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, "prepend_bos", False),
            prepend_lang_id=True,
        )

        self.datasets[split] = ds

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            src_tokens = sample["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            prefix_tokens = (
                torch.LongTensor([[self.tgt_dict.index("[{}]".format(self.args.target_lang))]]).expand(bsz, 1).to(src_tokens)
            )
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )
            

    def build_generator(self, models, args, **unused):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator
            
            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                temperature=getattr(args, "temperature", 1.0),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                #eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
                symbols_to_strip_from_output={self.source_dictionary.index("[{}]".format(self.args.target_lang))}
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        src_lang_id = self.source_dictionary.index("[{}]".format(self.args.source_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(
            source_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )
        return dataset
    
    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu
        def decode(toks):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                escape_unk=True, 
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )
            #if self.tokenizer:
            #    s = self.tokenizer.decode(s)
            return s
        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], self.target_dictionary.pad()
                )
            src_str = self.source_dictionary.string(src_tokens, 'sentencepiece')
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=gen_out[i][0]["tokens"],
                    src_str=src_str, 
                    alignment=gen_out[i][0]["alignment"],
                    align_dict=None,
                    tgt_dict=self.target_dictionary,
                    remove_bpe='sentencepiece',
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
            #detok_hypo_str = self.tokenizer.decode(hypo_str)
            hyps.append(hypo_str)
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    #escape_unk=False,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        return sacrebleu.corpus_bleu(hyps, [refs])
