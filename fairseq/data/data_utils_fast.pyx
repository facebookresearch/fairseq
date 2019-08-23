# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

cimport cython
cimport numpy as np

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t


cdef _is_batch_full(list batch, long num_tokens, long max_tokens, long max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


@cython.cdivision(True)
cpdef list batch_by_size_fast(
    np.ndarray[DTYPE_t, ndim=1] indices,
    num_tokens_fn,
    long max_tokens,
    long max_sentences,
    int bsz_mult,
):
    cdef long sample_len = 0
    cdef list sample_lens = []
    cdef list batch = []
    cdef list batches = []
    cdef long mod_len
    cdef long i
    cdef long idx
    cdef long num_tokens
    cdef DTYPE_t[:] indices_view = indices

    for i in range(len(indices_view)):
        idx = indices_view[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches
