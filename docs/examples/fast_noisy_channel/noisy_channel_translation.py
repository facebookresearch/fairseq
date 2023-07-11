# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq import checkpoint_utils
import argparse
from fairseq.tasks import register_task
import torch


@register_task("noisy_channel_translation")
class NoisyChannelTranslation(TranslationTask):
    """
    Rescore the top k candidates from each beam using noisy channel modeling
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        # fmt: off
        parser.add_argument('--channel-model', metavar='FILE',
                            help='path to P(S|T) model. P(S|T) and P(T|S) must share source and target dictionaries.')
        parser.add_argument('--combine-method', default='lm_only',
                            choices=['lm_only', 'noisy_channel'],
                            help="""method for combining direct and channel model scores.
                                    lm_only: decode with P(T|S)P(T)
                                    noisy_channel: decode with 1/t P(T|S) + 1/s(P(S|T)P(T))""")
        parser.add_argument('--normalize-lm-scores-by-tgt-len', action='store_true', default=False,
                            help='normalize lm score by target length instead of source length')
        parser.add_argument('--channel-scoring-type', default='log_norm', choices=['unnormalized', 'log_norm', 'k2_separate', 'src_vocab', 'src_vocab_batched'],
                            help="Normalize bw scores with log softmax or return bw scores without log softmax")
        parser.add_argument('--top-k-vocab', default=0, type=int,
                            help='top k vocab IDs to use with `src_vocab` in channel model scoring')
        parser.add_argument('--k2', default=50, type=int,
                            help='the top k2 candidates to rescore with the noisy channel model for each beam')
        parser.add_argument('--ch-wt', default=1, type=float,
                            help='weight for the channel model')
        parser.add_argument('--lm-model', metavar='FILE',
                            help='path to lm model file, to model P(T). P(T) must share the same vocab as the direct model on the target side')
        parser.add_argument('--lm-data', metavar='FILE',
                            help='path to lm model training data for target language, used to properly load LM with correct dictionary')
        parser.add_argument('--lm-wt', default=1, type=float,
                            help='the weight of the lm in joint decoding')
        # fmt: on

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if getattr(args, "score_reference", False):
            raise NotImplementedError()
        else:
            from .noisy_channel_sequence_generator import NoisyChannelSequenceGenerator
            use_cuda = torch.cuda.is_available() and not self.args.cpu
            assert self.args.lm_model is not None, '--lm-model required for noisy channel generation!'
            assert self.args.lm_data is not None, '--lm-data required for noisy channel generation to map between LM and bitext vocabs'
            if self.args.channel_model is not None:
                import copy
                ch_args_task = copy.deepcopy(self.args)
                tmp = ch_args_task.source_lang
                ch_args_task.source_lang = ch_args_task.target_lang
                ch_args_task.target_lang = tmp
                ch_args_task._name = 'translation'
                channel_task = TranslationTask.setup_task(ch_args_task)

            arg_dict = {}
            arg_dict['task'] = 'language_modeling'
            arg_dict['sample_break_mode'] = 'eos'
            arg_dict['data'] = self.args.lm_data
            arg_dict['output_dictionary_size'] = -1
            lm_args = argparse.Namespace(**arg_dict)
            lm_task = LanguageModelingTask.setup_task(lm_args)
            lm_dict = lm_task.output_dictionary

            if self.args.channel_model is not None:
                channel_models, _ = checkpoint_utils.load_model_ensemble(self.args.channel_model.split(':'), task=channel_task)

                for model in channel_models:
                    model.make_generation_fast_(
                        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                        need_attn=args.print_alignment,
                    )
                    if self.args.fp16:
                        model.half()
                    if use_cuda:
                        model.cuda()
            else:
                channel_models = None

            lm_models, _ = checkpoint_utils.load_model_ensemble(self.args.lm_model.split(':'), task=lm_task)

            for model in lm_models:
                model.make_generation_fast_(
                    beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                    need_attn=args.print_alignment,
                )
                if self.args.fp16:
                    model.half()
                if use_cuda:
                    model.cuda()
            return NoisyChannelSequenceGenerator(
                combine_method=self.args.combine_method,
                tgt_dict=self.target_dictionary,
                src_dict=self.source_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                channel_models=channel_models,
                k2=getattr(self.args, 'k2', 50),
                ch_weight=getattr(self.args, 'ch_wt', 1),
                channel_scoring_type=self.args.channel_scoring_type,
                top_k_vocab=self.args.top_k_vocab,
                lm_models=lm_models,
                lm_dict=lm_dict,
                lm_weight=getattr(self.args, 'lm_wt', 1),
                normalize_lm_scores_by_tgt_len=getattr(self.args, 'normalize_lm_scores_by_tgt_len', False),
            )
