# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import metrics

EVAL_BLEU_ORDER = 4


def maybe_compute_bleu(logging_outputs):

    has_bleu = False
    for log in logging_outputs:
        if not log.get("_bleu_sys_len") is None:
            has_bleu = True
            break

    if not has_bleu:
        return

    import numpy as np

    def sum_logs(key):
        import torch

        result = sum(log.get(key, 0) for log in logging_outputs)
        if torch.is_tensor(result):
            result = result.cpu().long().item()

        return result

    counts, totals = [], []
    for i in range(EVAL_BLEU_ORDER):
        counts.append(sum_logs("_bleu_counts_" + str(i)))
        totals.append(sum_logs("_bleu_totals_" + str(i)))

    if max(totals) > 0:
        # log counts as numpy arrays -- log_scalar will sum them correctly
        metrics.log_scalar("_bleu_counts", np.array(counts))
        metrics.log_scalar("_bleu_totals", np.array(totals))
        metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
        metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

        def compute_bleu(meters):
            import inspect

            try:
                from sacrebleu.metrics import BLEU

                comp_bleu = BLEU.compute_bleu
            except ImportError:
                # compatibility API for sacrebleu 1.x
                import sacrebleu

                comp_bleu = sacrebleu.compute_bleu

            fn_sig = inspect.getfullargspec(comp_bleu)[0]
            if "smooth_method" in fn_sig:
                smooth = {"smooth_method": "exp"}
            else:
                smooth = {"smooth": "exp"}
            bleu = comp_bleu(
                correct=meters["_bleu_counts"].sum,
                total=meters["_bleu_totals"].sum,
                sys_len=meters["_bleu_sys_len"].sum,
                ref_len=meters["_bleu_ref_len"].sum,
                **smooth,
            )
            return round(bleu.score, 2)

        metrics.log_derived("bleu", compute_bleu)
