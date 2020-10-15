# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from .loss.latent_depth import LatentLayersKLLoss, LatentLayersSparsityLoss


@register_task('multilingual_translation_latent_depth')
class MultilingualTranslationTaskLatentDepth(MultilingualTranslationTask):
    """A task for multiple translation with latent depth.

    See `"Deep Transformer with Latent Depth"
        (Li et al., 2020) <https://arxiv.org/pdf/2009.13102.pdf>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        MultilingualTranslationTask.add_args(parser)
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
        super().__init__(args, dicts, training)
        self.src_langs, self.tgt_langs = zip(*[(lang.split("-")[0], lang.split("-")[1]) for lang in args.lang_pairs])
        if self.training and self.encoder_latent_layer:
            assert self.args.share_encoders
        if self.training and self.decoder_latent_layer:
            assert self.args.share_decoders
        if training or self.encoder_latent_layer or self.decoder_latent_layer:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        self.eval_lang_pairs = self.lang_pairs
        self.model_lang_pairs = self.lang_pairs
        if self.training and (self.encoder_latent_layer or self.decoder_latent_layer):
            self.kl_loss = LatentLayersKLLoss(self.args)
            self.sparsity_loss = LatentLayersSparsityLoss(self.args)

    def _per_lang_pair_train_loss(self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad):
        src, tgt = lang_pair.split("-")
        if self.encoder_latent_layer:
            src_lang_idx = self.src_lang_idx_dict[src]
            model.models[lang_pair].encoder.set_lang_idx(src_lang_idx)
            model.models[lang_pair].encoder.layer_select.hard_select = update_num > self.args.soft_update
        if self.decoder_latent_layer:
            tgt_lang_idx = self.tgt_lang_idx_dict[tgt]
            model.models[lang_pair].decoder.set_lang_idx(tgt_lang_idx)
            model.models[lang_pair].decoder.layer_select.hard_select = update_num > self.args.soft_update

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

        if hasattr(self, "sparsity_loss") and self.sparsity_loss.is_valid(update_num):
            # need to retain the graph if sparsity loss needs to be added
            loss.backward(retain_graph=True)
        else:
            optimizer.backward(loss)

        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        agg_loss, agg_sample_size, agg_logging_output = super().train_step(
                sample, model, criterion, optimizer, update_num, ignore_grad)
        # compute auxiliary loss from layere sparsity, based on all samples from all languages
        if hasattr(self, "sparsity_loss") and self.sparsity_loss.is_valid(update_num):
            sparsity_loss = 0
            if self.encoder_latent_layer:
                sparsity_loss += self.sparsity_loss(
                        next(iter(model.models.values())).encoder.layer_select.layer_samples, update_num, agg_sample_size)
            if self.decoder_latent_layer:
                sparsity_loss += self.sparsity_loss(
                        next(iter(model.models.values())).decoder.layer_select.layer_samples, update_num, agg_sample_size)
            if sparsity_loss > 0:
                optimizer.backward(sparsity_loss)
        return agg_loss, agg_sample_size, agg_logging_output

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        src, tgt = lang_pair.split("-")
        if self.encoder_latent_layer:
            src_lang_idx = self.src_lang_idx_dict[src]
            model.models[lang_pair].encoder.set_lang_idx(src_lang_idx)
        if self.decoder_latent_layer:
            tgt_lang_idx = self.tgt_lang_idx_dict[tgt]
            model.models[lang_pair].decoder.set_lang_idx(tgt_lang_idx)
        loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
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
        return super().inference_step(generator, models, sample, prefix_tokens, constraints)

    @property
    def encoder_latent_layer(self):
        return hasattr(self.args, "encoder_latent_layer") and self.args.encoder_latent_layer

    @property
    def decoder_latent_layer(self):
        return hasattr(self.args, "decoder_latent_layer") and self.args.decoder_latent_layer

    @property
    def src_lang_idx_dict(self):
        return {lang: lang_idx for lang_idx, lang in enumerate(self.src_langs)}

    @property
    def tgt_lang_idx_dict(self):
        return {lang: lang_idx for lang_idx, lang in enumerate(self.tgt_langs)}
