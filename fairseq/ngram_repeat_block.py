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
    if not EXTENSION_BUILT or not torch.cuda.is_available():
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
    
    @staticmethod
    def custom_index_select(A: torch.Tensor, indx: torch.Tensor):
        """ Custom index_select implementation compatible with Torchscript """
        res = torch.reshape(A.index_select(0, indx[0][0])[0, indx[1][0]], (-1,)).to(A)
        for i in range(1, indx[0].size(0)):
            indx1 = indx[0][i]
            indx2 = indx[1][i]
            tmp = torch.reshape(A.index_select(0, indx1)[0, indx2], (-1,)).to(A)
            res = torch.cat((res, tmp), 0)
        final_shape = [-1]
        final_shape.extend(list(A.size()[2:]))
        res = torch.reshape(res, final_shape)
        return res

    @staticmethod
    def custom_index_put_(A: torch.Tensor, indx: torch.Tensor, value):
        """ Custom index_put_ implementation compatible with Torchscript """
        indices: List[torch.Tensor] = [i for i in indx]
        A.index_put_(indices=indices, values=value)
        return A

    def _no_repeat_ngram(
        self,
        tokens: torch.Tensor,
        lprobs: torch.Tensor,
        bsz: int,
        beam_size: int,
        step: int
    ) -> torch.Tensor:
        """For each hypothesis generate a list of previous ngrams and set associated lprobs to -inf"""
        
        num_ngrams = step - self.no_repeat_ngram_size + 2 # Number of ngrams generated so far

        if num_ngrams > 0:
            # Expand tokens tensor. The resulting gen_ngrams have shape (num_sequences, num_ngrams, ngram_size)
            expand_tokens = torch.unsqueeze(tokens, dim=1).clone().to(tokens)
            gen_ngrams = expand_tokens.to(tokens)
            for n in range(1, self.no_repeat_ngram_size):
                shift = torch.roll(expand_tokens, shifts=-n, dims=2)
                gen_ngrams = torch.cat((gen_ngrams, shift), dim=1)
            
            # Now we transpose gen_ngrams and truncate at the maximum number of ngrams generated so far
            gen_ngrams = torch.transpose(gen_ngrams[:, :, :num_ngrams ], dim0=2, dim1=1)

            # Keep the preceding tokens generated till step is reached on every batch/beam
            last_generated_ngram = tokens[:, step + 2 - self.no_repeat_ngram_size : step + 1]

            # Now we need to look for last_generated_ngram in our gen_ngrams matrix. 
            # All matches should return next token (the ones we need to avoid).
            expand_lgn = torch.unsqueeze(last_generated_ngram, dim=1).repeat(1, num_ngrams, 1)

            matches_mask = torch.all(gen_ngrams[:, :, :-1] == expand_lgn, dim=2).to(tokens)
            indices = torch.nonzero(matches_mask).to(tokens)

            #If there are no matches, then we can just return lprobs as it is
            if indices.size()[0] == 0:
                return lprobs
            
            # Tensor containing the tokens we need to avoid. 
            # Running "banned_tokens = gen_ngrams[indices][:, -1]" is not compatible with TS".
            banned_tokens = self.custom_index_select(gen_ngrams, indices.t())[:, -1]

            # Tensor to keep track of which batch/beam we are
            positions = torch.arange(bsz*beam_size).unsqueeze(-1).repeat(1, num_ngrams).to(tokens)

            #Where does the banned_tokens belong? Let's find out:
            selected_positions = self.custom_index_select(positions, indices.t())
            final_positions = torch.stack((selected_positions, banned_tokens), dim=1)
            final_positions = torch.unique(final_positions, dim=0)
            # Banned token positions will be set to -inf
            inf_value = torch.tensor(-float('inf')).to(lprobs)
            # Set to -inf those tokens that beam search should not generate next
            lprobs = self.custom_index_put_(lprobs, final_positions.t(), inf_value)

        return lprobs
