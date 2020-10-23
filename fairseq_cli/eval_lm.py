#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os
from argparse import Namespace

import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.dataclass.initialize import register_hydra_cfg
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from hydra.core.config_store import ConfigStore
from hydra.experimental import initialize
from omegaconf import DictConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
logger = logging.getLogger("fairseq_cli.eval_lm")


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """increments counters for the sum of log probs of current word and next
        word (given context ending at current word). Since the next word might be at the end of the example,
        or it might be not counted because it is not an ending subword unit,
        also keeps track of how many of those we have seen"""
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return "{}\t{}\t{}\t{}\t{}\t{}".format(
            self.word,
            self.count,
            self.log_prob,
            self.is_bpe,
            self.next_word_prob,
            self.count - self.missing_next_words,
        )


def main(cfg: DictConfig, override_args=None, **unused_kwargs):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    logger.info(cfg)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))

    # reduce tokens per sample by the required context window size
    cfg.task.tokens_per_sample -= cfg.eval_lm.context_window

    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Load dataset splits
    gen_subset = cfg.dataset.gen_subset
    task.load_dataset(gen_subset)
    dataset = task.dataset(gen_subset)
    if cfg.eval_lm.context_window > 0:
        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=cfg.task.tokens_per_sample,
            context_window=cfg.eval_lm.context_window,
            pad_idx=task.source_dictionary.pad(),
        )
    logger.info("{} {} {} examples".format(cfg.task.data, gen_subset, len(dataset)))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    assert len(models) > 0

    logger.info(
        "num. model params: {}".format(sum(p.numel() for p in models[0].parameters()))
    )

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens or 36000,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=True,
        num_shards=max(
            cfg.dataset.num_shards,
            cfg.distributed_training.distributed_world_size,
        ),
        shard_id=max(
            cfg.dataset.shard_id,
            cfg.distributed_training.distributed_rank,
        ),
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, cfg.eval_lm.softmax_batch)

    score_sum = 0.0
    count = 0

    if cfg.common_eval.post_process is not None:
        if cfg.common_eval.post_process == "sentencepiece":
            raise NotImplementedError
        else:
            bpe_cont = cfg.common_eval.post_process.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    wps_meter = TimeMeter()

    for sample in progress:
        if "net_input" not in sample:
            continue

        sample = utils.move_to_cuda(sample) if use_cuda else sample

        gen_timer.start()
        hypos = scorer.generate(models, sample)
        gen_timer.stop(sample["ntokens"])

        for i, hypos_i in enumerate(hypos):
            hypo = hypos_i[0]
            sample_id = sample["id"][i]

            tokens = hypo["tokens"]
            tgt_len = tokens.numel()
            pos_scores = hypo["positional_scores"].float()

            if cfg.task.add_bos_token:
                assert hypo["tokens"][0].item() == task.target_dictionary.bos()
                tokens = tokens[1:]
                pos_scores = pos_scores[1:]

            skipped_toks = 0
            if bpe_toks is not None:
                for i in range(tgt_len - 1):
                    if tokens[i].item() in bpe_toks:
                        skipped_toks += 1
                        pos_scores[i + 1] += pos_scores[i]
                        pos_scores[i] = 0

            inf_scores = pos_scores.eq(float("inf")) | pos_scores.eq(float("-inf"))
            if inf_scores.any():
                logger.info(
                    "skipping tokens with inf scores:",
                    task.target_dictionary.string(tokens[inf_scores.nonzero()]),
                )
                pos_scores = pos_scores[(~inf_scores).nonzero()]
            score_sum += pos_scores.sum().cpu()
            count += pos_scores.numel() - skipped_toks

            if cfg.eval_lm.output_word_probs or cfg.eval_lm.output_word_stats:
                w = ""
                word_prob = []
                is_bpe = False
                for i in range(len(tokens)):
                    w_ind = tokens[i].item()
                    w += task.source_dictionary[w_ind]
                    if bpe_toks is not None and w_ind in bpe_toks:
                        w = w[:-bpe_len]
                        is_bpe = True
                    else:
                        word_prob.append((w, pos_scores[i].item()))

                        next_prob = None
                        ind = i + 1
                        while ind < len(tokens):
                            if pos_scores[ind].item() != 0:
                                next_prob = pos_scores[ind]
                                break
                            ind += 1

                        word_stats.setdefault(w, WordStat(w, is_bpe)).add(
                            pos_scores[i].item(), next_prob
                        )
                        is_bpe = False
                        w = ""
                if cfg.eval_lm.output_word_probs:
                    logger.info(
                        str(int(sample_id))
                        + " "
                        + (
                            "\t".join(
                                "{} [{:2f}]".format(x[0], x[1]) for x in word_prob
                            )
                        )
                    )

        wps_meter.update(sample["ntokens"])
        progress.log({"wps": round(wps_meter.avg)})

    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
    logger.info(
        "Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)".format(
            gen_timer.n, gen_timer.sum, 1.0 / gen_timer.avg
        )
    )
    logger.info(
        "Loss (base 2): {:.4f}, Perplexity: {:.2f}".format(
            avg_nll_loss, 2 ** avg_nll_loss
        )
    )

    if cfg.eval_lm.output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            logger.info(ws)


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    register_hydra_cfg(cs)
    initialize(config_path="../config", strict=True)
    cli_main()
