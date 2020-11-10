#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Flashlight decoders.
"""

import gc
import itertools as it
import os.path as osp
import warnings
from collections import deque, namedtuple

import numpy as np
import torch
from examples.speech_recognition.data.replabels import unpack_replabels
from fairseq import tasks
from fairseq.utils import apply_to_sample
from omegaconf import open_dict
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


try:
    from flashlight.lib.text.dictionary import create_word_dict, load_words
    from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder,
    )
except:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object


class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        if args.criterion == "ctc":
            self.criterion_type = CriterionType.CTC
            self.blank = (
                tgt_dict.index("<ctc_blank>")
                if "<ctc_blank>" in tgt_dict.indices
                else tgt_dict.bos()
            )
            self.asg_transitions = None
        elif args.criterion == "asg_loss":
            self.criterion_type = CriterionType.ASG
            self.blank = -1
            self.asg_transitions = args.asg_transitions
            self.max_replabel = args.max_replabel
            assert len(self.asg_transitions) == self.vocab_size ** 2
        else:
            raise RuntimeError(f"unknown criterion: {args.criterion}")

    def generate(self, models, sample, **unused):
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
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = models[0](**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)
        elif self.criterion_type == CriterionType.ASG:
            emissions = encoder_out["encoder_out"]
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        if self.criterion_type == CriterionType.CTC:
            idxs = filter(lambda x: x != self.blank, idxs)
        elif self.criterion_type == CriterionType.ASG:
            idxs = filter(lambda x: x >= 0, idxs)
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

        self.silence = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        self.lexicon = load_words(args.lexicon)
        self.word_dict = create_word_dict(self.lexicon)
        self.unk_word = self.word_dict.get_index("<unk>")

        self.lm = KenLM(args.kenlm_model, self.word_dict)
        self.trie = Trie(self.vocab_size, self.silence)

        start_state = self.lm.start(False)
        for i, (word, spellings) in enumerate(self.lexicon.items()):
            word_idx = self.word_dict.get_index(word)
            _, score = self.lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idxs = [tgt_dict.index(token) for token in spelling]
                assert (
                    tgt_dict.unk() not in spelling_idxs
                ), f"{spelling} {spelling_idxs}"
                self.trie.insert(spelling_idxs, word_idx, score)
        self.trie.smear(SmearingMode.MAX)

        self.decoder_opts = LexiconDecoderOptions(
            beam_size=args.beam,
            beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
            beam_threshold=args.beam_threshold,
            lm_weight=args.lm_weight,
            word_score=args.word_score,
            unk_score=args.unk_weight,
            sil_score=args.sil_weight,
            log_add=False,
            criterion_type=self.criterion_type,
        )


        if self.asg_transitions is None:
            N = 768
            # self.asg_transitions = torch.FloatTensor(N, N).zero_()
            self.asg_transitions = []

        self.decoder = LexiconDecoder(
            self.decoder_opts,
            self.trie,
            self.lm,
            self.silence,
            self.blank,
            self.unk_word,
            self.asg_transitions,
            False,
        )

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(result.tokens),
                        "score": result.score,
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    }
                    for result in nbest_results
                ]
            )
        return hypos


FairseqLMState = namedtuple("FairseqLMState", ["prefix", "incremental_state", "probs"])


class FairseqLM(LM):
    def __init__(self, dictionary, model):
        LM.__init__(self)
        self.dictionary = dictionary
        self.model = model
        self.unk = self.dictionary.unk()

        self.save_incremental = False  # this currently does not work properly
        self.max_cache = 20_000

        model.cuda()
        model.eval()
        model.make_generation_fast_()

        self.states = {}
        self.stateq = deque()

    def start(self, start_with_nothing):
        state = LMState()
        prefix = torch.LongTensor([[self.dictionary.eos()]])
        incremental_state = {} if self.save_incremental else None
        with torch.no_grad():
            res = self.model(prefix.cuda(), incremental_state=incremental_state)
            probs = self.model.get_normalized_probs(res, log_probs=True, sample=None)

        if incremental_state is not None:
            incremental_state = apply_to_sample(lambda x: x.cpu(), incremental_state)
        self.states[state] = FairseqLMState(
            prefix.numpy(), incremental_state, probs[0, -1].cpu().numpy()
        )
        self.stateq.append(state)

        return state

    def score(self, state: LMState, token_index: int, no_cache: bool = False):
        """
        Evaluate language model based on the current lm state and new word
        Parameters:
        -----------
        state: current lm state
        token_index: index of the word
                     (can be lexicon index then you should store inside LM the
                      mapping between indices of lexicon and lm, or lm index of a word)

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        curr_state = self.states[state]

        def trim_cache(targ_size):
            while len(self.stateq) > targ_size:
                rem_k = self.stateq.popleft()
                rem_st = self.states[rem_k]
                rem_st = FairseqLMState(rem_st.prefix, None, None)
                self.states[rem_k] = rem_st

        if curr_state.probs is None:
            new_incremental_state = (
                curr_state.incremental_state.copy()
                if curr_state.incremental_state is not None
                else None
            )
            with torch.no_grad():
                if new_incremental_state is not None:
                    new_incremental_state = apply_to_sample(
                        lambda x: x.cuda(), new_incremental_state
                    )
                elif self.save_incremental:
                    new_incremental_state = {}

                res = self.model(
                    torch.from_numpy(curr_state.prefix).cuda(),
                    incremental_state=new_incremental_state,
                )
                probs = self.model.get_normalized_probs(
                    res, log_probs=True, sample=None
                )

                if new_incremental_state is not None:
                    new_incremental_state = apply_to_sample(
                        lambda x: x.cpu(), new_incremental_state
                    )

                curr_state = FairseqLMState(
                    curr_state.prefix, new_incremental_state, probs[0, -1].cpu().numpy()
                )

            if not no_cache:
                self.states[state] = curr_state
                self.stateq.append(state)

        score = curr_state.probs[token_index].item()

        trim_cache(self.max_cache)

        outstate = state.child(token_index)
        if outstate not in self.states and not no_cache:
            prefix = np.concatenate(
                [curr_state.prefix, torch.LongTensor([[token_index]])], -1
            )
            incr_state = curr_state.incremental_state

            self.states[outstate] = FairseqLMState(prefix, incr_state, None)

        if token_index == self.unk:
            score = float("-inf")

        return outstate, score

    def finish(self, state: LMState):
        """
        Evaluate eos for language model based on the current lm state

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        return self.score(state, self.dictionary.eos())

    def empty_cache(self):
        self.states = {}
        self.stateq = deque()
        gc.collect()


class W2lFairseqLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.silence = tgt_dict.bos()

        self.unit_lm = getattr(args, "unit_lm", False)

        self.lexicon = load_words(args.lexicon) if args.lexicon else None
        self.idx_to_wrd = {}

        checkpoint = torch.load(args.kenlm_model, map_location="cpu")

        if "cfg" in checkpoint and checkpoint["cfg"] is not None:
            lm_args = checkpoint["cfg"]
        else:
            lm_args = convert_namespace_to_omegaconf(checkpoint["args"])

        with open_dict(lm_args.task):
            lm_args.task.data = osp.dirname(args.kenlm_model)

        task = tasks.setup_task(lm_args.task)
        model = task.build_model(lm_args.model)
        model.load_state_dict(checkpoint["model"], strict=False)

        self.trie = Trie(self.vocab_size, self.silence)

        self.word_dict = task.dictionary
        self.unk_word = self.word_dict.unk()
        self.lm = FairseqLM(self.word_dict, model)

        self.decoder_opts = LexiconDecoderOptions(
            beam_size=args.beam,
            beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
            beam_threshold=args.beam_threshold,
            lm_weight=args.lm_weight,
            word_score=args.word_score,
            unk_score=args.unk_weight,
            sil_score=args.sil_weight,
            log_add=False,
            criterion_type=self.criterion_type,
        )

        if self.lexicon:
            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                if self.unit_lm:
                    word_idx = i
                    self.idx_to_wrd[i] = word
                    score = 0
                else:
                    word_idx = self.word_dict.index(word)
                    _, score = self.lm.score(start_state, word_idx, no_cache=True)

                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]
                    assert (
                        tgt_dict.unk() not in spelling_idxs
                    ), f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                [],
                self.unit_lm,
            )
        else:
            from flashlight.lib.text.decoder import LexiconFreeDecoder
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []

        def idx_to_word(idx):
            if self.unit_lm:
                return self.idx_to_wrd[idx]
            else:
                return self.word_dict[idx]

        def make_hypo(result):
            hypo = {"tokens": self.get_tokens(result.tokens), "score": result.score}
            if self.lexicon:
                hypo["words"] = [idx_to_word(x) for x in result.words if x >= 0]
            return hypo

        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append([make_hypo(result) for result in nbest_results])
            self.lm.empty_cache()

        return hypos
