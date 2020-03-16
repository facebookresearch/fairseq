"""
Scores references and hypos
"""
from abc import abstractmethod
import logging
from typing import *

from fairseq import registry, metrics
from fairseq.utils import item
import numpy as np


logger = logging.getLogger(__name__)


build_eval_scorer, register_eval_scorer, EVAL_SCORER_REGISTRY = registry.setup_registry(
    '--eval-scorer',
    default=None,
)


def add_eval_scoring_args(parser):
    parser.add_argument('--eval-scorer-args', type=str, metavar='JSON',
                        help='generation args for validation inference, '
                             'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
    parser.add_argument('--eval-scorer-detok', type=str, default="space",
                        help='detokenizer before computing any scores (e.g., "moses"); '
                             'required if using --eval-bleu; use "space" to '
                             'disable detokenization; see fairseq.data.encoders '
                             'for other options')
    parser.add_argument('--eval-scorer-detok-args', type=str, metavar='JSON',
                        help='args for building the tokenizer, if needed')
    parser.add_argument('--eval-scorer-remove-bpe', nargs='?', const='@@ ', default=None,
                        help='remove BPE before computing any scores')
    parser.add_argument('--eval-scorer-print-samples', action='store_true', default=False,
                        help='print sample generations during validation')
    parser.add_argument('--eval-scorer-lowercase', action='store_true', default=True,
                        help='lowercase outputs/references before computing any score')


class GenerationScorer:
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def score(self, references, hypos) -> Dict[str, Any]:
        """
        Score references and hypos and output a dictionary that can be included in logging_outputs.
        """
        pass

    @abstractmethod
    def reduce_metrics(self, logging_outputs, criterion) -> None:
        """ Reduce added fields in logging_outputs into a single metric and log it to metrics. """
        pass


@register_eval_scorer("eval-bleu")
class BleuGenerationScorer(GenerationScorer):
    """ Most of it moved from Translation. """

    @staticmethod
    def add_args(parser):
        # options for reporting BLEU during validation
        parser.add_argument('--eval-scorer-bleu-tokenized', action='store_true', default=True,
                            help='if setting, we compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-scorer-bleu-smooth', choices=['exp', 'floor', 'add-k', 'none'], default='exp',
                            help='smoothing method: exponential decay (default), floor (increment zero counts), add-k (increment num/denom by k for n>1), or none')
        parser.add_argument('--eval-scorer-bleu-smooth-value', type=float, default=0.,
                            help='The value to pass to the smoothing technique, when relevant. Default: %(default)s.')

    def __init__(self, args):
        super().__init__(args)
        self.eval_bleu_order = 4
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("Please install sacrebleu with pip install sacrebleu.")

    def score(self, refs: Iterable[str], hypos: Iterable[str]) -> Dict:
        import sacrebleu
        tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args.eval_scorer_bleu_tokenized else 'none'
        bleu = sacrebleu.corpus_bleu(
            hypos,
            [refs],
            tokenize=tokenize,
        )

        # create logging output in here
        assert len(bleu.counts) == self.eval_bleu_order
        bleu_logging_output = {
            '_bleu_sys_len': bleu.sys_len,
            '_bleu_ref_len': bleu.ref_len,
        }
        # we split counts into separate entries so that they can be
        # summed efficiently across workers using fast-stat-sync
        for i in range(self.eval_bleu_order):
            bleu_logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
            bleu_logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]

        return bleu_logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        def sum_logs(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        counts, totals = [], []
        for i in range(self.eval_bleu_order):
            counts.append(sum_logs('_bleu_counts_' + str(i)))
            totals.append(sum_logs('_bleu_totals_' + str(i)))

        if max(totals) > 0:
            # log counts as numpy arrays -- log_scalar will sum them correctly
            metrics.log_scalar('_bleu_counts', np.array(counts))
            metrics.log_scalar('_bleu_totals', np.array(totals))
            metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
            metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

            def compute_bleu(meters):
                import inspect
                import sacrebleu
                fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                if 'smooth_method' in fn_sig:
                    smooth = {'smooth_method': self.args.eval_scorer_bleu_smooth}
                else:
                    smooth = {'smooth': self.args.eval_scorer_bleu_smooth}

                bleu = sacrebleu.compute_bleu(
                    correct=meters['_bleu_counts'].sum,
                    total=meters['_bleu_totals'].sum,
                    sys_len=meters['_bleu_sys_len'].sum,
                    ref_len=meters['_bleu_ref_len'].sum,
                    smooth_value=self.args.eval_scorer_bleu_smooth_value,
                    **smooth
                )
                return round(bleu.score, 2)

            metrics.log_derived('bleu', compute_bleu)


@register_eval_scorer("eval-precision-recall")
class PrecisionRecallGenerationScorer(GenerationScorer):
    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def precision(tp: int, fp: int) -> float:
        return tp / (tp + fp) if (tp + fp) > 0 else 0.

    @staticmethod
    def recall(tp: int, fn: int) -> float:
        if (tp + fn) == 0:
            return 0.
        return float(tp) / (tp + fn)

    @staticmethod
    def f_measure(tp: int, fp: int, fn: int, beta=1) -> float:
        f_precision = PrecisionRecallGenerationScorer.precision(tp, fp)
        f_recall = PrecisionRecallGenerationScorer.recall(tp, fn)
        if f_precision == 0 or f_recall == 0:
            return 0.
        else:
            return (1 + beta ** 2) * (f_precision * f_recall) / ((beta ** 2 * f_precision) + f_recall)

    @staticmethod
    def sentence_score(
            reference: Iterable[Any], hypothesis: Iterable[Any]
    ) -> Tuple[int, int, int]:

        tp, fp, fn = 0, 0, 0

        reference_set = set(reference)
        hypothesis_set = set(hypothesis)

        for token in hypothesis:
            if token in reference_set:
                tp += 1
            else:
                fp += 1

        for token in reference:
            if token not in hypothesis_set:
                fn += 1

        return tp, fp, fn

    @staticmethod
    def corpus_macro_score(
            refs: Iterable[Iterable[Any]], hypos: Iterable[Iterable[Any]]
    ) -> Tuple[int, int, int]:
        tp, fp, fn = 0, 0, 0
        for ref, hypo in zip(refs, hypos):
            sample_tp, sample_fp, sample_fn = PrecisionRecallGenerationScorer.sentence_score(ref, hypo)
            tp += sample_tp
            fp += sample_fp
            fn += sample_fn
        return tp, fp, fn

    # implement scorer interface below
    def score(self, refs, hypos) -> Dict:
        assert len(refs) == len(hypos)

        total = len(refs)
        refs = (ref.split() for ref in refs)  # tokenize words FIXME maybe use a tokenizer
        hypos = (hypo.split() for hypo in hypos)

        tp, fp, fn = self.corpus_macro_score(refs, hypos)

        precision_logging_output = {
            '_pr_tp': tp,
            '_pr_fp': fp,
            '_pr_fn': fn,
            '_pr_total': total
        }

        return precision_logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        def sum_logs(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        total = sum_logs('_pr_total')
        if total:
            metrics.log_scalar('_pr_tp', sum_logs('_pr_tp'))
            metrics.log_scalar('_pr_fp', sum_logs('_pr_fp'))
            metrics.log_scalar('_pr_fn', sum_logs('_pr_fn'))

            def compute_precision(meters):
                _precision = self.precision(
                    tp=meters['_pr_tp'].sum,
                    fp=meters['_pr_fp'].sum,
                )
                return round(item(_precision * 100), 2)

            def compute_recall(meters):
                _recall = self.recall(
                    tp=meters['_pr_tp'].sum,
                    fn=meters['_pr_fn'].sum,
                )
                return round(item(_recall * 100), 2)

            def compute_f_measure(meters):
                _f_measure = self.f_measure(
                    tp=meters['_pr_tp'].sum,
                    fp=meters['_pr_fp'].sum,
                    fn=meters['_pr_fn'].sum,
                )

                return round(item(_f_measure * 100), 2)

            metrics.log_derived('recall', compute_recall)
            metrics.log_derived('precision', compute_precision)
            metrics.log_derived('f-1', compute_f_measure)
