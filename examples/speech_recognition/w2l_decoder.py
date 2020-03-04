#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wav2letter decoders.
"""
import math
import itertools as it
import torch
from fairseq import utils
from examples.speech_recognition.data.replabels import unpack_replabels
from wav2letter.common import create_word_dict, load_words
from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
from wav2letter.decoder import (
    CriterionType,
    DecoderOptions,
    KenLM,
    SmearingMode,
    Trie,
    WordLMDecoder,
)


class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        if args.criterion == "ctc_loss":
            self.criterion_type = CriterionType.CTC
            self.blank = tgt_dict.index("<ctc_blank>")
            self.asg_transitions = None
        elif args.criterion == "asg_loss":
            self.criterion_type = CriterionType.ASG
            self.blank = -1
            self.asg_transitions = args.asg_transitions
            self.max_replabel = args.max_replabel
            assert len(self.asg_transitions) == self.vocab_size ** 2
        else:
            raise RuntimeError(f"unknown criterion: {args.criterion}")

    def generate(self, models, sample, prefix_tokens=None):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        encoder_out = models[0].encoder(**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)
        elif self.criterion_type == CriterionType.ASG:
            emissions = encoder_out["encoder_out"]
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x >= 0, idxs)
        if self.criterion_type == CriterionType.CTC:
            idxs = filter(lambda x: x != self.blank, idxs)
        elif self.criterion_type == CriterionType.ASG:
            idxs = unpack_replabels(list(idxs), self.tgt_dict, self.max_replabel)
        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
            for b in range(B)
        ]


class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.silence = tgt_dict.index(args.silence_token)

        self.lexicon = load_words(args.lexicon)
        self.word_dict = create_word_dict(self.lexicon)
        self.unk_word = self.word_dict.get_index("<unk>")

        self.lm = KenLM(args.kenlm_model, self.word_dict)
        self.trie = Trie(self.vocab_size, self.silence)

        start_state = self.lm.start(False)
        for word, spellings in self.lexicon.items():
            word_idx = self.word_dict.get_index(word)
            _, score = self.lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idxs = [tgt_dict.index(token) for token in spelling]
                self.trie.insert(spelling_idxs, word_idx, score)
        self.trie.smear(SmearingMode.MAX)

        self.decoder_opts = DecoderOptions(
            args.beam,
            args.beam_threshold,
            args.lm_weight,
            args.word_score,
            args.unk_weight,
            False,
            args.sil_weight,
            self.criterion_type,
        )

        self.decoder = WordLMDecoder(
            self.decoder_opts,
            self.trie,
            self.lm,
            self.silence,
            self.blank,
            self.unk_word,
            self.asg_transitions,
        )

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            nbest_results = self.decoder.decode(emissions_ptr, T, N)[: self.nbest]
            hypos.append(
                [
                    {"tokens": self.get_tokens(result.tokens), "score": result.score}
                    for result in nbest_results
                ]
            )
        return hypos
