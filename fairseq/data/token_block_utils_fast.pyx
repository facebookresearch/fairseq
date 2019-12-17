# cython: language_level=3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from itertools import chain
from libc.math cimport ceil

cimport cython
cimport numpy as np

DTYPE = np.int64
ctypedef np.int64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_t, ndim=2] _get_slice_indices_none_mode(np.ndarray[DTYPE_t, ndim=1] sizes, int block_size):
    cdef DTYPE_t total_size = sizes.sum()
    cdef DTYPE_t length = <DTYPE_t> ceil(total_size / <double> block_size)
    cdef np.ndarray[DTYPE_t, ndim=2] slice_indices = np.zeros([length, 2], dtype=DTYPE)
    cdef DTYPE_t[:, :] slice_indices_view = slice_indices
    cdef DTYPE_t i
    cdef DTYPE_t start
    cdef DTYPE_t end
    for i in range(length):
        start = i * block_size
        end = min(start + block_size, total_size)
        slice_indices_view[i][0] = start
        slice_indices_view[i][1] = end
    return slice_indices


cdef np.ndarray[DTYPE_t, ndim=2] _fast_convert_to_np_array(list list_of_list):
    """
    Faster function to convert DTYPE_t list of list.
    Only fast when there are huge number of rows and low number of columns.
    """
    cdef np.ndarray[DTYPE_t, ndim=1] flat = np.fromiter(chain.from_iterable(list_of_list), DTYPE, -1)
    return flat.reshape((len(list_of_list), -1))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[DTYPE_t, ndim=2] _get_slice_indices_fast(np.ndarray[DTYPE_t, ndim=1] sizes, str break_mode, int block_size, int document_sep_len):
    cdef DTYPE_t tok_idx = 0
    cdef DTYPE_t sz_idx = 0
    cdef DTYPE_t curr_size = 0
    cdef DTYPE_t i = 0
    cdef DTYPE_t length
    cdef DTYPE_t total_size
    cdef DTYPE_t[:] sizes_view = sizes
    cdef np.ndarray[DTYPE_t, ndim=2] slice_indices
    cdef list slice_indices_list = []

    if break_mode is None or break_mode == 'none':
        slice_indices = _get_slice_indices_none_mode(sizes, block_size)
    elif break_mode == 'complete':
        while sz_idx < len(sizes_view):
            if curr_size + sizes_view[sz_idx] <= block_size or curr_size == 0:
                curr_size += sizes_view[sz_idx]
                sz_idx += 1
            else:
                slice_indices_list.append((tok_idx, tok_idx + curr_size))
                tok_idx += curr_size
                curr_size = 0
        if curr_size > 0:
            slice_indices_list.append((tok_idx, tok_idx + curr_size))
        slice_indices = _fast_convert_to_np_array(slice_indices_list)
    elif break_mode == 'complete_doc':
        while sz_idx < len(sizes_view):
            if (
                (curr_size + sizes_view[sz_idx] <= block_size or curr_size == 0)
                # an empty sentence indicates end-of-document:
                and sizes_view[sz_idx] != document_sep_len
            ):
                curr_size += sizes_view[sz_idx]
                sz_idx += 1
            else:
                # Only keep non-empty documents.
                if curr_size > 1:
                    slice_indices_list.append((tok_idx, tok_idx + curr_size))
                tok_idx += curr_size
                curr_size = 0
                if sizes_view[sz_idx] == document_sep_len:
                    tok_idx += sizes_view[sz_idx]
                    sz_idx += 1
        if curr_size > 1:
            slice_indices_list.append((tok_idx, tok_idx + curr_size))
        slice_indices = _fast_convert_to_np_array(slice_indices_list)
    elif break_mode == 'eos':
        slice_indices = np.zeros((len(sizes), 2), dtype=DTYPE)
        cumsum = sizes.cumsum(axis=0)
        slice_indices[1:, 0] = cumsum[:cumsum.shape[0] - 1]
        slice_indices[:, 1] = cumsum
    else:
        raise ValueError('Invalid break_mode: ' + break_mode)
    return slice_indices


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[DTYPE_t, ndim=2] _get_block_to_dataset_index_fast(np.ndarray[DTYPE_t, ndim=1] sizes, np.ndarray[DTYPE_t, ndim=2] slice_indices):
    cdef DTYPE_t start_ds_idx
    cdef DTYPE_t start_offset
    cdef DTYPE_t end_ds_idx
    cdef DTYPE_t i
    cdef DTYPE_t s
    cdef DTYPE_t e
    cdef DatasetSearcher ds = DatasetSearcher(sizes)
    cdef np.ndarray[DTYPE_t, ndim=2] block_to_dataset_index = np.zeros([len(slice_indices), 3], dtype=DTYPE)
    cdef DTYPE_t[:, :] block_to_dataset_index_view = block_to_dataset_index
    cdef DTYPE_t[:, :] slice_indices_view = slice_indices
    cdef Py_ssize_t x_max = slice_indices.shape[0]

    for i in range(x_max):
        s = slice_indices_view[i][0]
        e = slice_indices_view[i][1]
        ds.seek(s)
        start_ds_idx = ds.current_index
        start_offset = ds.current_offset
        if e <= s:
            end_ds_idx = start_ds_idx
        else:
            ds.seek(e - 1)
            end_ds_idx = ds.current_index
        block_to_dataset_index_view[i][0] = start_ds_idx  # starting index in dataset
        block_to_dataset_index_view[i][1] = start_offset  # starting offset within starting index
        block_to_dataset_index_view[i][2] = end_ds_idx    # ending index in dataset
    return block_to_dataset_index


cdef class DatasetSearcher(object):
    """Helper for mapping "flat" indices to indices and offsets in an
    underlying dataset."""
    cdef DTYPE_t current_i
    cdef DTYPE_t current_offset
    cdef DTYPE_t current_index
    cdef DTYPE_t[:] sizes

    def __init__(self, DTYPE_t[:] sizes):
        self.sizes = sizes
        self.reset()

    cdef reset(self):
        self.current_offset = 0     # offset within current index in underlying dataset
        self.current_i = 0          # "flat" index
        self.current_index = 0      # index in underlying dataset

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef int step(self, DTYPE_t i):
        cdef DTYPE_t to_consume
        cdef DTYPE_t remaining
        if i < self.current_i:
            self.reset()
        if i > self.current_i:
            to_consume = i - self.current_i
            remaining = self.sizes[self.current_index] - self.current_offset
            if remaining > to_consume:
                self.current_offset += to_consume
                self.current_i += to_consume
            else:
                assert remaining > 0
                self.current_i += remaining
                self.current_index += 1
                self.current_offset = 0
                return 1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef seek(self, DTYPE_t i):
        cdef int not_done = 1
        while not_done == 1:
            not_done = self.step(i)
        assert self.current_i == i
