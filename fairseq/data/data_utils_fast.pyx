# cython: language_level=3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

cimport cython
cimport numpy as np

from libc.stdint cimport int32_t, int64_t

ctypedef int64_t DTYPE_t


cdef _is_batch_full(int64_t num_sentences, int64_t num_tokens, int64_t max_tokens, int64_t max_sentences):
    if num_sentences == 0:
        return 0
    if max_sentences > 0 and num_sentences == max_sentences:
        return 1
    if max_tokens > 0 and num_tokens > max_tokens:
        return 1
    return 0


@cython.cdivision(True)
cpdef list batch_by_size_fast(
    np.ndarray[DTYPE_t, ndim=1] indices,
    num_tokens_fn,
    int64_t max_tokens,
    int64_t max_sentences,
    int32_t bsz_mult,
):
    cdef int64_t sample_len = 0
    cdef list sample_lens = []
    cdef list batch = []
    cdef list batches = []
    cdef int64_t mod_len
    cdef int64_t i
    cdef int64_t idx
    cdef int64_t num_tokens
    cdef DTYPE_t[:] indices_view = indices

    for i in range(len(indices_view)):
        idx = indices_view[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert max_tokens <= 0 or sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(len(batch), num_tokens, max_tokens, max_sentences):
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


cdef _find_valid_shape(
    DTYPE_t[:, :] shapes_view,
    int64_t num_sentences,
    int64_t num_tokens,
):
    """Return index of first valid shape of -1 if none is found."""
    for i in range(shapes_view.shape[0]):
        if num_sentences <= shapes_view[i][0] and num_tokens <= shapes_view[i][1]:
            return i
    return -1


@cython.cdivision(True)
cpdef list batch_fixed_shapes_fast(
    np.ndarray[DTYPE_t, ndim=1] indices,
    num_tokens_fn,
    np.ndarray[DTYPE_t, ndim=2] fixed_shapes_sorted,
):
    cdef int64_t sample_len = 0
    cdef list sample_lens = []
    cdef list batch = []
    cdef list batches = []
    cdef int64_t mod_len
    cdef int64_t i
    cdef int64_t idx
    cdef int64_t num_tokens
    cdef DTYPE_t[:] indices_view = indices
    cdef DTYPE_t[:, :] shapes_view = fixed_shapes_sorted

    for i in range(len(indices_view)):
        idx = indices_view[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        shape_idx = _find_valid_shape(shapes_view, len(batch) + 1, sample_len)
        if shape_idx == -1:
            batches.append(batch)
            batch = []
            sample_lens = []
            sample_len = 0
            shapes_view = fixed_shapes_sorted
        elif shape_idx > 0:
            # small optimization for the next call to _find_valid_shape
            shapes_view = shapes_view[shape_idx:]

        batch.append(idx)

    if len(batch) > 0:
        batches.append(batch)

    return batches
