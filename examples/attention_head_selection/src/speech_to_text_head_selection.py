# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask

from .data.speech_to_text_dataset_with_domain import SpeechToTextDatasetCreatorWithDomain
from .loss.attention_head_selection import HeadSelectionLoss


@register_task("speech_to_text_head_selection")
class SpeechToTextHeadSelectionTask(SpeechToTextTask):

    @classmethod
    def add_args(cls, parser):
        SpeechToTextTask.add_args(parser)
        parser.add_argument(
            "--task-type",
            type=str,
            default="lang",
            help="task type for head selection, lang or domain"
        )
        parser.add_argument(
            "--kl-weight",
            type=float,
            default=0.0,
            help="the weight of KL loss"
        )

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        self.task_type = args.task_type
        assert self.task_type in ["lang", "domain"], "invalid task_type: {}, should be either lang or domain".format(self.task_type)
        self.map_task_to_id(args.train_subset)
        self.encoder_head_prior = float(args.decoder_attention_heads) / args.total_decoder_attention_heads
        self.decoder_head_prior = float(args.encoder_attention_heads) / args.total_encoder_attention_heads
        self.kl_loss = HeadSelectionLoss(args)

    def map_task_to_id(self, train_subset):
        src_lang_set, tgt_lang_set, domain_set = set(), set(), set()
        for split in train_subset.split(","):
            seq = split.split("_")
            assert len(seq) == 4, "subset {} should be in the format of train_src_tgt_domain".format(split)
            _, src_lang, tgt_lang, domain = seq
            src_lang_set.add(src_lang)
            tgt_lang_set.add(tgt_lang)
            domain_set.add(domain)
        src_langs = sorted(src_lang_set)
        tgt_langs = sorted(tgt_lang_set)
        domains = sorted(domain_set)
        self.src_lang_map = {src_lang: i for (i, src_lang) in enumerate(src_langs)}
        self.tgt_lang_map = {tgt_lang: i for (i, tgt_lang) in enumerate(tgt_langs)}
        self.domain_map = {domain: i for (i, domain) in enumerate(domains)}
        if self.task_type == "lang":
            self.encoder_tasks = len(self.src_lang_map)
            self.decoder_tasks = len(self.tgt_lang_map)
        elif self.task_type == "domain":
            self.encoder_tasks = len(self.domain_map)
            self.decoder_tasks = len(self.domain_map)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreatorWithDomain.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            src_lang_map=self.src_lang_map,
            tgt_lang_map=self.tgt_lang_map,
            domain_map=self.domain_map,
            speaker_to_id=self.speaker_to_id
        )

    def build_model(self, args):
        args.encoder_tasks = self.encoder_tasks
        args.decoder_tasks = self.decoder_tasks
        return super(SpeechToTextHeadSelectionTask, self).build_model(args)

    def get_sample_sizes(self, sample, task_ids, num_tasks):
        """
        task_ids: (bsz,)
        get sample sizes for each task
        """
        bsz = task_ids.size(0)
        mat = torch.zeros((num_tasks, bsz), device=task_ids.device)
        mat[task_ids, torch.arange(bsz)] = 1.0
        ntokens = torch.sum(sample['target'] != 1, dim=-1)
        sample_sizes = torch.matmul(mat, ntokens.float())
        return sample_sizes

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        # task ids
        if self.task_type == "lang":
            encoder_task_ids = sample["src_lang_ids"]
            decoder_task_ids = sample["tgt_lang_ids"]
        elif self.task_type == "domain":
            encoder_task_ids = sample["domain_ids"]
            decoder_task_ids = sample["domain_ids"]
        model.encoder.set_task_ids(encoder_task_ids)
        model.decoder.set_task_ids(decoder_task_ids)

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
                # KL loss
                if self.args.encoder_attn_head_select:
                    sample_sizes = self.get_sample_sizes(sample, encoder_task_ids, self.encoder_tasks)
                    loss += self.kl_loss(
                        model.encoder.attn_head_selector.head_samples,
                        sample_sizes,
                        self.encoder_head_prior
                    )
                if self.args.decoder_self_attn_head_select:
                    sample_sizes = self.get_sample_sizes(sample, decoder_task_ids, self.decoder_tasks)
                    loss += self.kl_loss(
                        model.decoder.self_attn_head_selector.head_samples,
                        sample_sizes,
                        self.decoder_head_prior
                    )
                if self.args.dec_enc_attn_head_select:
                    sample_sizes = self.get_sample_sizes(sample, decoder_task_ids, self.decoder_tasks)
                    loss += self.kl_loss(
                        model.decoder.enc_attn_head_selector.head_sampes,
                        sample_sizes,
                        self.decoder_head_prior
                    )

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        # task ids
        if self.task_type == "lang":
            encoder_task_ids = sample["src_lang_ids"]
            decoder_task_ids = sample["tgt_lang_ids"]
        elif self.task_type == "domain":
            encoder_task_ids = sample["domain_ids"]
            decoder_task_ids = sample["domain_ids"]
        model.encoder.set_task_ids(encoder_task_ids)
        model.decoder.set_task_ids(decoder_task_ids)
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # task ids
            if self.task_type == "lang":
                encoder_task_ids = sample["src_lang_ids"][:1]
                decoder_task_ids = sample["tgt_lang_ids"][:1]
            elif self.task_type == "domain":
                encoder_task_ids = sample["domain_ids"][:1]
                decoder_task_ids = sample["domain_ids"][:1]
            for model in models:
                model.encoder.set_task_ids(encoder_task_ids)
                model.decoder.set_task_ids(decoder_task_ids)
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )
