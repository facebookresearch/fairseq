#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import itertools as it
import math
import os.path as osp
import warnings
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from examples.speech_recognition.data.replabels import unpack_replabels
from fairseq import tasks
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models.fairseq_model import FairseqModel
from fairseq.utils import apply_to_sample
from omegaconf import MISSING, open_dict

try:
    from flashlight.lib.sequence.criterion import (CpuViterbiPath,
                                                   get_data_ptr_as_bytes)
    from flashlight.lib.text.decoder import (LM, CriterionType, DecodeResult,
                                             KenLM, LexiconDecoder,
                                             LexiconDecoderOptions,
                                             LexiconFreeDecoder,
                                             LexiconFreeDecoderOptions,
                                             LMState, SmearingMode, Trie)
    from flashlight.lib.text.dictionary import create_word_dict, load_words
except ImportError:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. "
        "Please install from "
        "https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object


CRITERION_CHOICES = ChoiceEnum(["ctc", "asg"])
DECODER_CHOICES = ChoiceEnum(["viterbi", "kenlm", "fairseqlm"])


@dataclass
class DecoderConfig(FairseqDataclass):
    name: DECODER_CHOICES = field(
        default="viterbi",
        metadata={"help": "The type of decoder to use"},
    )
    nbest: int = field(
        default=1,
        metadata={"help": "Number of decodings to return"},
    )
    criterion: CRITERION_CHOICES = field(
        default="ctc",
        metadata={"help": "Criterion to use"},
    )
    asgtransitions: List[int] = field(
        default=MISSING,
        metadata={"help": "ASG transition indices"},
    )
    maxreplabel: int = field(
        default=2,
        metadata={"help": "Maximum repeated labels for ASG criterion"},
    )
    unitlm: bool = field(
        default=False,
        metadata={"help": "If set, use unit language model"},
    )
    lmpath: str = field(
        default=MISSING,
        metadata={"help": "Language model for KenLM decoder"},
    )
    lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "Lexicon for Flashlight decoder"},
    )
    beam: int = field(
        default=50,
        metadata={"help": "Number of beams to use for decoding"},
    )
    beamthreshold: float = field(
        default=15.0,
        metadata={"help": "Threshold for beam search decoding"},
    )
    beamsizetoken: Optional[int] = field(
        default=None,
        metadata={"help": "Beam size to use"}
    )
    wordscore: float = field(
        default=1.5,
        metadata={"help": "Word score for KenLM decoder"},
    )
    unkweight: float = field(
        default=-math.inf,
        metadata={"help": "Unknown weight for KenLM decoder"},
    )
    silweight: float = field(
        default=-0.3,
        metadata={"help": "Silence weight for KenLM decoder"},
    )
    lmweight: float = field(
        default=1.5,
        metadata={"help": "Weight for LM while interpolating score"},
    )
    unitlm: bool = field(
        default=False,
        metadata={"help": "If using a unit language model"},
    )


class BaseDecoder:
    def __init__(self, cfg: DecoderConfig, tgt_dict: Dictionary) -> None:
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = cfg.nbest
        self.unitlm = cfg.unitlm

        if cfg.criterion == "ctc":
            self.criterion_type = CriterionType.CTC
            self.blank = (
                tgt_dict.index("<ctc_blank>")
                if "<ctc_blank>" in tgt_dict.indices
                else tgt_dict.bos()
            )
            if "<sep>" in tgt_dict.indices:
                self.silence = tgt_dict.index("<sep>")
            elif "|" in tgt_dict.indices:
                self.silence = tgt_dict.index("|")
            else:
                self.silence = tgt_dict.eos()
            self.asgtransitions = None
        elif cfg.criterion == "asg_loss":
            self.criterion_type = CriterionType.ASG
            self.blank = -1
            self.silence = -1
            self.asgtransitions = cfg.asgtransitions
            self.maxreplabel = cfg.maxreplabel
            assert len(self.asgtransitions) == self.vocab_size ** 2
        else:
            raise RuntimeError(f"unknown criterion: {cfg.criterion}")

    def generate(
        self,
        models: List[FairseqModel],
        sample: Dict[str, Any],
        **unused
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        encoder_input = {
            k: v
            for k, v in sample["net_input"].items()
            if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(
        self,
        models: List[FairseqModel],
        encoder_input: Dict[str, Any],
    ) -> torch.FloatTensor:
        model = models[0]
        encoder_out = model(**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(encoder_out)
            else:
                emissions = model.get_normalized_probs(
                    encoder_out, log_probs=True)
        elif self.criterion_type == CriterionType.ASG:
            emissions = encoder_out["encoder_out"]
        else:
            raise ValueError("Criterion not implemented: "
                             f"{self.criterion_type}")
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs: torch.IntTensor) -> torch.LongTensor:
        idxs = (g[0] for g in it.groupby(idxs))
        if self.criterion_type == CriterionType.CTC:
            idxs = filter(lambda x: x != self.blank, idxs)
        elif self.criterion_type == CriterionType.ASG:
            idxs = filter(lambda x: x >= 0, idxs)
            idxs = unpack_replabels(
                list(idxs), self.tgt_dict, self.maxreplabel)
        return torch.LongTensor(list(idxs))

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        raise NotImplementedError


class ViterbiDecoder(BaseDecoder):
    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        B, T, N = emissions.size()
        if self.asgtransitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asgtransitions).view(N, N)
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(
            CpuViterbiPath.get_workspace_size(B, T, N))
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


class KenLMDecoder(BaseDecoder):
    def __init__(self, cfg: DecoderConfig, tgt_dict: Dictionary) -> None:
        super().__init__(cfg, tgt_dict)

        if cfg.lexicon:
            self.lexicon = load_words(cfg.lexicon)
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")

            self.lm = KenLM(cfg.lmpath, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)

            start_state = self.lm.start(False)
            for word, spellings in self.lexicon.items():
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    spelling_idxs = [
                        tgt_dict.index(token)
                        for token in spelling
                    ]
                    assert tgt_dict.unk() not in spelling_idxs, \
                        f"{word} {spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=cfg.beam,
                beam_size_token=cfg.beamsizetoken or len(tgt_dict),
                beam_threshold=cfg.beamthreshold,
                lm_weight=cfg.lmweight,
                word_score=cfg.wordscore,
                unk_score=cfg.unkweight,
                sil_score=cfg.silweight,
                log_add=False,
                criterion_type=self.criterion_type,
            )

            if self.asgtransitions is None:
                self.asgtransitions = []

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                self.asgtransitions,
                self.unitlm,
            )
        else:
            assert self.unitlm, "Lexicon-free decoding requires unit LM"

            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(cfg.lmpath, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=cfg.beam,
                beam_size_token=cfg.beamsizetoken or len(tgt_dict),
                beam_threshold=cfg.beamthreshold,
                lm_weight=cfg.lmweight,
                sil_score=cfg.silweight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append([
                {
                    "tokens": self.get_tokens(result.tokens),
                    "score": result.score,
                    "words": [
                        self.word_dict.get_entry(x)
                        for x in result.words if x >= 0
                    ],
                } for result in nbest_results
            ])
        return hypos


FairseqLMState = namedtuple(
    "FairseqLMState",
    [
        "prefix",
        "incremental_state",
        "probs",
    ]
)


class FairseqLM(LM):
    def __init__(self, dictionary: Dictionary, model: FairseqModel) -> None:
        super().__init__()

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

    def start(self, start_with_nothing: bool) -> LMState:
        state = LMState()
        prefix = torch.LongTensor([[self.dictionary.eos()]])
        incremental_state = {} if self.save_incremental else None
        with torch.no_grad():
            res = self.model(
                prefix.cuda(), incremental_state=incremental_state)
            probs = self.model.get_normalized_probs(
                res, log_probs=True, sample=None)

        if incremental_state is not None:
            incremental_state = apply_to_sample(
                lambda x: x.cpu(), incremental_state)
        self.states[state] = FairseqLMState(
            prefix.numpy(), incremental_state, probs[0, -1].cpu().numpy()
        )
        self.stateq.append(state)

        return state

    def score(
        self,
        state: LMState,
        token_index: int,
        no_cache: bool = False,
    ) -> Tuple[LMState, int]:
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

        def trim_cache(targ_size: int) -> None:
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
                    curr_state.prefix, new_incremental_state, probs[0, -1].cpu(
                    ).numpy()
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

    def finish(self, state: LMState) -> Tuple[LMState, int]:
        """
        Evaluate eos for language model based on the current lm state
        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        return self.score(state, self.dictionary.eos())

    def empty_cache(self) -> None:
        self.states = {}
        self.stateq = deque()
        gc.collect()


class FairseqLMDecoder(BaseDecoder):
    def __init__(self, cfg: DecoderConfig, tgt_dict: Dictionary) -> None:
        super().__init__(cfg, tgt_dict)

        self.lexicon = load_words(cfg.lexicon) if cfg.lexicon else None
        self.idx_to_wrd = {}

        checkpoint = torch.load(cfg.lmpath, map_location="cpu")

        if "cfg" in checkpoint and checkpoint["cfg"] is not None:
            lm_args = checkpoint["cfg"]
        else:
            lm_args = convert_namespace_to_omegaconf(checkpoint["args"])

        with open_dict(lm_args.task):
            lm_args.task.data = osp.dirname(cfg.lmpath)

        task = tasks.setup_task(lm_args.task)
        model = task.build_model(lm_args.model)
        model.load_state_dict(checkpoint["model"], strict=False)

        self.trie = Trie(self.vocab_size, self.silence)

        self.word_dict = task.dictionary
        self.unk_word = self.word_dict.unk()
        self.lm = FairseqLM(self.word_dict, model)

        if self.lexicon:
            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                if self.unitlm:
                    word_idx = i
                    self.idx_to_wrd[i] = word
                    score = 0
                else:
                    word_idx = self.word_dict.index(word)
                    _, score = self.lm.score(
                        start_state, word_idx, no_cache=True)

                for spelling in spellings:
                    spelling_idxs = [
                        tgt_dict.index(token)
                        for token in spelling
                    ]
                    assert tgt_dict.unk() not in spelling_idxs, \
                        f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=cfg.beam,
                beam_size_token=cfg.beamsizetoken or len(tgt_dict),
                beam_threshold=cfg.beamthreshold,
                lm_weight=cfg.lmweight,
                word_score=cfg.wordscore,
                unk_score=cfg.unkweight,
                sil_score=cfg.silweight,
                log_add=False,
                criterion_type=self.criterion_type,
            )

            if self.asgtransitions is None:
                self.asgtransitions = []

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                self.asgtransitions,
                self.unitlm,
            )
        else:
            assert self.unitlm, "Lexicon-free decoding requires unit LM"

            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(cfg.lmpath, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=cfg.beam,
                beam_size_token=cfg.beamsizetoken or len(tgt_dict),
                beam_threshold=cfg.beamthreshold,
                lm_weight=cfg.lmweight,
                sil_score=cfg.silweight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        B, T, N = emissions.size()
        hypos = []

        def make_hypo(result: DecodeResult) -> Dict[str, Any]:
            hypo = {
                "tokens": self.get_tokens(result.tokens),
                "score": result.score,
            }
            if self.lexicon:
                hypo["words"] = [
                    self.idx_to_wrd[x] if self.unitlm else self.word_dict[x]
                    for x in result.words if x >= 0
                ]
            return hypo

        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[:self.nbest]
            hypos.append([make_hypo(result) for result in nbest_results])
            self.lm.empty_cache()

        return hypos


def Decoder(cfg: DecoderConfig, tgt_dict: Dictionary) -> BaseDecoder:
    if cfg.name == "viterbi":
        return ViterbiDecoder(cfg, tgt_dict)
    if cfg.name == "kenlm":
        return KenLMDecoder(cfg, tgt_dict)
    if cfg.name == "fairseqlm":
        return FairseqLMDecoder(cfg, tgt_dict)
    raise NotImplementedError(f"Invalid decoder name: {cfg.name}")
