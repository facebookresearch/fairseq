# Originally from Microsoft Corporation.
# Licensed under the MIT License.

""" Wrapper for ngram_repeat_block cuda extension """
import torch
from torch import nn

import math
from typing import Dict, List, Optional
import warnings

try:
    from fairseq import ngram_repeat_block_cuda

    EXTENSION_BUILT = True
except ImportError:
    EXTENSION_BUILT = False


def is_cuda_extension_usable() -> bool:
    """Check whether ngram_repeat_block_cuda is built properly"""
    if not EXTENSION_BUILT:
        return False
    bsz = 2
    tokens = torch.tensor([[4, 4, 3, 2], [1, 2, 3, 4]], dtype=torch.long, device="cuda")
    lprobs = torch.rand((8, 12), device="cuda")
    try:
        outputs = ngram_repeat_block_cuda.forward(tokens, lprobs, bsz, 3, 4, 3)
        outputs = outputs + 4  # This line breaks if the extension is built incorrectly.
        return True
    except RuntimeError:
        warnings.warn(
            "NGramRepeatBlock extension must be rebuilt."
            'Run TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0" python setup.py build_ext --inplace'
        )
        return False


class NGramRepeatBlock(nn.Module):
    """ Wrapper class for calling ngram_repeat_block cuda extension """

    def __init__(self, no_repeat_ngram_size: int, use_extension: bool = True):
        super().__init__()
        self.use_extension = is_cuda_extension_usable() if use_extension else False
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def reset_parameters(self):
        pass

    @torch.jit.unused
    def call_cuda_extension(
        self,
        tokens,
        lprobs,
        bsz: int,
        beam_size: int,
        step: int,
    ):
        return ngram_repeat_block_cuda.forward(
            tokens, lprobs, bsz, step, beam_size, self.no_repeat_ngram_size
        )

    def forward(
        self,
        tokens,
        lprobs,
        bsz: int,
        beam_size: int,
        step: int,
    ):
        """
        Args:
            tokens(Tensor): Input tokens(Bsz*beam, seq_len)
            lprobs(Tensor): likelihood probability,
            Expected to be updated in place.(Bsz*beam, vocab_size)
            bsz(int): batch size
            step(int): current step
            beam_size(int): beam size
            no_repeat_ngram_size(int): Ngram size
        """
        msg = f"expected {bsz *beam_size} got"
        assert tokens.size(0) == bsz * beam_size, f"{msg} {tokens.size(0)}"
        assert lprobs.size(0) == bsz * beam_size, f"{msg} {lprobs.size(0)}"
        if self.use_extension:
            return self.call_cuda_extension(tokens, lprobs, bsz, beam_size, step)

        else:
            return self._no_repeat_ngram(
                tokens,
                lprobs,
                bsz,
                beam_size,
                step,
            )

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        """For each hypothesis generate a list of previous ngrams and set associated lprobs to -inf"""
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {})
            for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list(
                [gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
            ):
                key = ",".join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
                    key, torch.jit.annotate(List[int], [])
                ) + [ngram[-1]]
        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(
                    tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
                )
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
            ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx]).long()
            ] = torch.tensor(-math.inf).to(lprobs)
        return lprobs

    @staticmethod
    def calculate_banned_tokens(
        tokens,
        step: int,
        gen_ngrams: List[Dict[str, List[int]]],
        no_repeat_ngram_size: int,
        bbsz_idx: int,
    ):
        tokens_list: List[int] = tokens[
            bbsz_idx, step + 2 - no_repeat_ngram_size : step + 1
        ].tolist()
        # before decoding the next token, prevent decoding of ngrams that have already appeared
        ngram_index = ",".join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    @staticmethod
    def transpose_list(l: List[List[int]]):
        # GeneratorExp aren't supported in TS so ignoring the lint
        min_len = min([len(x) for x in l])  # noqa
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2
