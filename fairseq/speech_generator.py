# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig


class SpeechGenerator(object):
    def __init__(self, model, vocoder, data_cfg: S2TDataConfig):
        self.model = model
        self.vocoder = vocoder
        stats_npz_path = data_cfg.global_cmvn_stats_npz
        self.gcmvn_stats = None
        if stats_npz_path is not None:
            self.gcmvn_stats = np.load(stats_npz_path)

    def gcmvn_denormalize(self, x):
        # x: B x T x C
        if self.gcmvn_stats is None:
            return x
        mean = torch.from_numpy(self.gcmvn_stats["mean"]).to(x)
        std = torch.from_numpy(self.gcmvn_stats["std"]).to(x)
        assert len(x.shape) == 3 and mean.shape[0] == std.shape[0] == x.shape[2]
        x = x * std.view(1, 1, -1).expand_as(x)
        return x + mean.view(1, 1, -1).expand_as(x)

    def get_waveform(self, feat):
        # T x C -> T
        return None if self.vocoder is None else self.vocoder(feat).squeeze(0)


class AutoRegressiveSpeechGenerator(SpeechGenerator):
    def __init__(
        self,
        model,
        vocoder,
        data_cfg,
        max_iter: int = 6000,
        eos_prob_threshold: float = 0.5,
    ):
        super().__init__(model, vocoder, data_cfg)
        self.max_iter = max_iter
        self.eos_prob_threshold = eos_prob_threshold

    @torch.no_grad()
    def generate(self, model, sample, has_targ=False, **kwargs):
        model.eval()

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()[:2]
        n_frames_per_step = model.decoder.n_frames_per_step
        out_dim = model.decoder.out_dim
        raw_dim = out_dim // n_frames_per_step

        # initialize
        encoder_out = model.forward_encoder(
            src_tokens, src_lengths, speaker=sample["speaker"]
        )
        incremental_state = {}
        feat, attn, eos_prob = [], [], []
        finished = src_tokens.new_zeros((bsz,)).bool()
        out_lens = src_lengths.new_zeros((bsz,)).long().fill_(self.max_iter)

        prev_feat_out = encoder_out["encoder_out"][0].new_zeros(bsz, 1, out_dim)
        for step in range(self.max_iter):
            cur_out_lens = out_lens.clone()
            cur_out_lens.masked_fill_(cur_out_lens.eq(self.max_iter), step + 1)
            _, cur_eos_out, cur_extra = model.forward_decoder(
                prev_feat_out,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                target_lengths=cur_out_lens,
                speaker=sample["speaker"],
                **kwargs,
            )
            cur_eos_prob = torch.sigmoid(cur_eos_out).squeeze(2)
            feat.append(cur_extra["feature_out"])
            attn.append(cur_extra["attn"])
            eos_prob.append(cur_eos_prob)

            cur_finished = cur_eos_prob.squeeze(1) > self.eos_prob_threshold
            out_lens.masked_fill_((~finished) & cur_finished, step + 1)
            finished = finished | cur_finished
            if finished.sum().item() == bsz:
                break
            prev_feat_out = cur_extra["feature_out"]

        feat = torch.cat(feat, dim=1)
        feat = model.decoder.postnet(feat) + feat
        eos_prob = torch.cat(eos_prob, dim=1)
        attn = torch.cat(attn, dim=2)
        alignment = attn.max(dim=1)[1]

        feat = feat.reshape(bsz, -1, raw_dim)
        feat = self.gcmvn_denormalize(feat)

        eos_prob = eos_prob.repeat_interleave(n_frames_per_step, dim=1)
        attn = attn.repeat_interleave(n_frames_per_step, dim=2)
        alignment = alignment.repeat_interleave(n_frames_per_step, dim=1)
        out_lens = out_lens * n_frames_per_step

        finalized = [
            {
                "feature": feat[b, :out_len],
                "eos_prob": eos_prob[b, :out_len],
                "attn": attn[b, :, :out_len],
                "alignment": alignment[b, :out_len],
                "waveform": self.get_waveform(feat[b, :out_len]),
            }
            for b, out_len in zip(range(bsz), out_lens)
        ]

        if has_targ:
            assert sample["target"].size(-1) == out_dim
            tgt_feats = sample["target"].view(bsz, -1, raw_dim)
            tgt_feats = self.gcmvn_denormalize(tgt_feats)
            tgt_lens = sample["target_lengths"] * n_frames_per_step
            for b, (f, l) in enumerate(zip(tgt_feats, tgt_lens)):
                finalized[b]["targ_feature"] = f[:l]
                finalized[b]["targ_waveform"] = self.get_waveform(f[:l])
        return finalized


class MultiDecoderSpeechGenerator(SpeechGenerator):
    def __init__(
        self,
        models,
        args,
        vocoder,
        data_cfg,
        tgt_dict_mt,
        max_iter: int = 6000,
        eos_prob_threshold: float = 0.5,
        eos_mt=None,
        symbols_to_strip_from_output=None,
    ):
        super().__init__(models[0], vocoder, data_cfg)
        self.max_iter = max_iter
        self.eos_prob_threshold = eos_prob_threshold

        self.tgt_dict_mt = tgt_dict_mt
        self.eos_mt = eos_mt

        from examples.speech_to_speech.unity.sequence_generator import SequenceGenerator
        from fairseq import search

        self.text_generator = SequenceGenerator(
            models,
            tgt_dict_mt,
            beam_size=max(1, getattr(args, "beam", 5)),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search.BeamSearch(tgt_dict_mt),
            eos=eos_mt,
            symbols_to_strip_from_output=symbols_to_strip_from_output,
        )

    @torch.no_grad()
    def generate(self, model, sample, has_targ=False, **kwargs):
        model.eval()

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()[:2]
        n_frames_per_step = model.decoder.n_frames_per_step
        out_dim = model.decoder.out_dim
        raw_dim = out_dim // n_frames_per_step

        # initialize
        encoder_out = model.forward_encoder(
            src_tokens, src_lengths, speaker=sample["speaker"]
        )

        prefix_tokens = None
        constraints = None
        bos_token = None

        mt_decoder = getattr(model, f"{model.mt_task_name}_decoder")

        # 1. MT decoder
        finalized_mt = self.text_generator.generate_decoder(
            [encoder_out],
            src_tokens,
            src_lengths,
            sample,
            prefix_tokens,
            constraints,
            bos_token,
            aux_task_name=model.mt_task_name,
        )

        # extract decoder output corresponding to the best hypothesis
        max_tgt_len = max([len(hypo[0]["tokens"]) for hypo in finalized_mt])
        prev_output_tokens_mt = (
            src_tokens.new_zeros(src_tokens.shape[0], max_tgt_len)
            .fill_(mt_decoder.padding_idx)
            .int()
        )  # B x T
        for i, hypo in enumerate(finalized_mt):
            i_beam = 0
            tmp = hypo[i_beam]["tokens"].int()  # hyp + eos
            prev_output_tokens_mt[i, 0] = self.text_generator.eos
            if tmp[-1] == self.text_generator.eos:
                tmp = tmp[:-1]
            prev_output_tokens_mt[i, 1 : len(tmp) + 1] = tmp

            text = "".join([self.tgt_dict_mt[c] for c in tmp])
            text = text.replace("_", " ")
            text = text.replace("▁", " ")
            text = text.replace("<unk>", " ")
            text = text.replace("<s>", "")
            text = text.replace("</s>", "")
            if len(text) > 0 and text[0] == " ":
                text = text[1:]
            sample_id = sample["id"].tolist()[i]
            print("{} (None-{})".format(text, sample_id))

        mt_decoder_out = mt_decoder(
            prev_output_tokens_mt,
            encoder_out=encoder_out,
            features_only=True,
        )
        x = mt_decoder_out[0].transpose(0, 1)

        mt_decoder_padding_mask = None
        if prev_output_tokens_mt.eq(mt_decoder.padding_idx).any():
            mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)

        # 2. TTS encoder
        if getattr(model, "synthesizer_encoder", None) is not None:
            synthesizer_encoder_out = model.synthesizer_encoder(
                x,
                mt_decoder_padding_mask,
            )
        else:
            synthesizer_encoder_out = {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [mt_decoder_padding_mask]
                if mt_decoder_padding_mask is not None
                else [],  # B x T
                "encoder_embedding": [],
                "encoder_states": [],
                "src_tokens": [],
                "src_lengths": [],
            }

        # 3. TTS decoder
        incremental_state = {}
        feat, attn, eos_prob = [], [], []
        finished = src_tokens.new_zeros((bsz,)).bool()
        out_lens = src_lengths.new_zeros((bsz,)).long().fill_(self.max_iter)

        prev_feat_out = encoder_out["encoder_out"][0].new_zeros(bsz, 1, out_dim)
        for step in range(self.max_iter):
            cur_out_lens = out_lens.clone()
            cur_out_lens.masked_fill_(cur_out_lens.eq(self.max_iter), step + 1)
            _, cur_eos_out, cur_extra = model.forward_decoder(
                prev_feat_out,
                encoder_out=synthesizer_encoder_out,
                incremental_state=incremental_state,
                target_lengths=cur_out_lens,
                speaker=sample["speaker"],
                **kwargs,
            )
            cur_eos_prob = torch.sigmoid(cur_eos_out).squeeze(2)
            feat.append(cur_extra["feature_out"])
            attn.append(cur_extra["attn"])
            eos_prob.append(cur_eos_prob)

            cur_finished = cur_eos_prob.squeeze(1) > self.eos_prob_threshold
            out_lens.masked_fill_((~finished) & cur_finished, step + 1)
            finished = finished | cur_finished
            if finished.sum().item() == bsz:
                break
            prev_feat_out = cur_extra["feature_out"]

        feat = torch.cat(feat, dim=1)
        feat = model.decoder.postnet(feat) + feat
        eos_prob = torch.cat(eos_prob, dim=1)
        attn = torch.cat(attn, dim=2)
        alignment = attn.max(dim=1)[1]

        feat = feat.reshape(bsz, -1, raw_dim)
        feat = self.gcmvn_denormalize(feat)

        eos_prob = eos_prob.repeat_interleave(n_frames_per_step, dim=1)
        attn = attn.repeat_interleave(n_frames_per_step, dim=2)
        alignment = alignment.repeat_interleave(n_frames_per_step, dim=1)
        out_lens = out_lens * n_frames_per_step

        finalized = [
            {
                "feature": feat[b, :out_len],
                "eos_prob": eos_prob[b, :out_len],
                "attn": attn[b, :, :out_len],
                "alignment": alignment[b, :out_len],
                "waveform": self.get_waveform(feat[b, :out_len]),
            }
            for b, out_len in zip(range(bsz), out_lens)
        ]

        if has_targ:
            assert sample["target"].size(-1) == out_dim
            tgt_feats = sample["target"].view(bsz, -1, raw_dim)
            tgt_feats = self.gcmvn_denormalize(tgt_feats)
            tgt_lens = sample["target_lengths"] * n_frames_per_step
            for b, (f, l) in enumerate(zip(tgt_feats, tgt_lens)):
                finalized[b]["targ_feature"] = f[:l]
                finalized[b]["targ_waveform"] = self.get_waveform(f[:l])
        return finalized


class NonAutoregressiveSpeechGenerator(SpeechGenerator):
    @torch.no_grad()
    def generate(self, model, sample, has_targ=False, **kwargs):
        model.eval()

        bsz, max_src_len = sample["net_input"]["src_tokens"].size()
        n_frames_per_step = model.encoder.n_frames_per_step
        out_dim = model.encoder.out_dim
        raw_dim = out_dim // n_frames_per_step

        feat, feat_post, out_lens, log_dur_out, _, _ = model(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            incremental_state=None,
            target_lengths=sample["target_lengths"],
            speaker=sample["speaker"],
        )
        if feat_post is not None:
            feat = feat_post

        feat = feat.view(bsz, -1, raw_dim)
        feat = self.gcmvn_denormalize(feat)

        dur_out = torch.clamp(torch.round(torch.exp(log_dur_out) - 1).long(), min=0)

        def get_dur_plot_data(d):
            r = []
            for i, dd in enumerate(d):
                r += [i + 1] * dd.item()
            return r

        out_lens = out_lens * n_frames_per_step
        finalized = [
            {
                "feature": feat[b, :l] if l > 0 else feat.new_zeros([1, raw_dim]),
                "waveform": self.get_waveform(
                    feat[b, :l] if l > 0 else feat.new_zeros([1, raw_dim])
                ),
                "attn": feat.new_tensor(get_dur_plot_data(dur_out[b])),
            }
            for b, l in zip(range(bsz), out_lens)
        ]

        if has_targ:
            tgt_feats = sample["target"].view(bsz, -1, raw_dim)
            tgt_feats = self.gcmvn_denormalize(tgt_feats)
            tgt_lens = sample["target_lengths"] * n_frames_per_step
            for b, (f, l) in enumerate(zip(tgt_feats, tgt_lens)):
                finalized[b]["targ_feature"] = f[:l]
                finalized[b]["targ_waveform"] = self.get_waveform(f[:l])
        return finalized


class TeacherForcingAutoRegressiveSpeechGenerator(AutoRegressiveSpeechGenerator):
    @torch.no_grad()
    def generate(self, model, sample, has_targ=False, **kwargs):
        model.eval()

        src_tokens = sample["net_input"]["src_tokens"]
        src_lens = sample["net_input"]["src_lengths"]
        prev_out_tokens = sample["net_input"]["prev_output_tokens"]
        tgt_lens = sample["target_lengths"]
        n_frames_per_step = model.decoder.n_frames_per_step
        raw_dim = model.decoder.out_dim // n_frames_per_step
        bsz = src_tokens.shape[0]

        feat, eos_prob, extra = model(
            src_tokens,
            src_lens,
            prev_out_tokens,
            incremental_state=None,
            target_lengths=tgt_lens,
            speaker=sample["speaker"],
        )

        attn = extra["attn"]  # B x T_s x T_t
        alignment = attn.max(dim=1)[1]
        feat = feat.reshape(bsz, -1, raw_dim)
        feat = self.gcmvn_denormalize(feat)
        eos_prob = eos_prob.repeat_interleave(n_frames_per_step, dim=1)
        attn = attn.repeat_interleave(n_frames_per_step, dim=2)
        alignment = alignment.repeat_interleave(n_frames_per_step, dim=1)
        tgt_lens = sample["target_lengths"] * n_frames_per_step

        finalized = [
            {
                "feature": feat[b, :tgt_len],
                "eos_prob": eos_prob[b, :tgt_len],
                "attn": attn[b, :, :tgt_len],
                "alignment": alignment[b, :tgt_len],
                "waveform": self.get_waveform(feat[b, :tgt_len]),
            }
            for b, tgt_len in zip(range(bsz), tgt_lens)
        ]

        if has_targ:
            tgt_feats = sample["target"].view(bsz, -1, raw_dim)
            tgt_feats = self.gcmvn_denormalize(tgt_feats)
            for b, (f, l) in enumerate(zip(tgt_feats, tgt_lens)):
                finalized[b]["targ_feature"] = f[:l]
                finalized[b]["targ_waveform"] = self.get_waveform(f[:l])
        return finalized
