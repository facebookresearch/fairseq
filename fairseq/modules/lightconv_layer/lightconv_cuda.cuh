/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <stdlib.h>
#include <assert.h>

#define SHFL_MASK 0xffffffff

template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void lightconv_forward_kernel(const scalar_t* input,
                              const scalar_t* filters,
                              int minibatch, int sequenceLength,
                              int numFeatures, int numFiltersInBlock,
                              scalar_t* output);

template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void lightconv_grad_wrt_input_kernel(
    const scalar_t* input, 
    const scalar_t* filters,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    scalar_t* output);

template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void lightconv_grad_wrt_weights_firstpass_short_kernel(
    const scalar_t* input,
    const scalar_t* gradInput,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    int numHeads,
    float* output);

template<int FS, int SB, typename scalar_t>
__global__
void lightconv_grad_wrt_weights_secondpass_short_kernel(
    const float* input,
    const int minibatch, 
    const int numFiltersInBlock,
    scalar_t* output);

template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void lightconv_grad_wrt_weights_firstpass_kernel(
    const scalar_t* input,
    const scalar_t* gradInput,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    float* output);

template<int FS, int SB, typename scalar_t>
__global__
void lightconv_grad_wrt_weights_secondpass_kernel(
    const float* input,
    const int minibatch, 
    const int numFiltersInBlock,
    scalar_t* output);

