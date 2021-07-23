/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "../cuda_utils.cu"
#include "lightconv_cuda.cuh"
#include "lightconv_cuda_backward.cu"
#include "lightconv_cuda_forward.cu"

template <int FS, int SB, int padding_l, typename scalar_t>
__global__ void lightconv_forward_kernel(
    const scalar_t* input,
    const scalar_t* filters,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    scalar_t* output) {
  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int featureIdx = blockIdx.y;
  const int filterIdx = featureIdx / numFiltersInBlock;

  const int IOOffset =
      numFeatures * sequenceLength * batchIdx + featureIdx * sequenceLength;
  const scalar_t* inputFeature = &input[IOOffset];
  scalar_t* outputFeature = &output[IOOffset];
  const scalar_t* inputFilter = &filters[filterIdx * FS];

  assert(blockDim.x == SB);

  scalar_t filter[FS];
#pragma unroll
  for (int i = 0; i < FS; ++i) {
    filter[i] = inputFilter[i];
  }

  __shared__ scalar_t temp[SB + FS];
  zeroSharedMem<FS, SB, padding_l>(temp);

  const int numIterations = divUp<int, int>(sequenceLength, SB);

  for (int i = 0; i < numIterations; ++i) {
    // Read input into shared memory
    const int inputOffset = i * SB;

    load_input_to_shared<FS, SB, padding_l>(
        inputFeature,
        inputOffset,
        sequenceLength,
        i,
        numIterations,
        (numIterations == 1),
        temp);

    __syncthreads();

    scalar_t out = 0;
#pragma unroll
    for (int j = 0; j < FS; ++j) {
      out += filter[j] * temp[tid + j];
    }

    // Write output
    const int outputOffset = inputOffset;
    if ((outputOffset + tid) < sequenceLength) {
      outputFeature[outputOffset + tid] = out;
    }

    __syncthreads();
  }
}

template <int FS, int SB, int padding_l, typename scalar_t>
__global__ void lightconv_grad_wrt_input_kernel(
    const scalar_t* input,
    const scalar_t* filters,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    scalar_t* output) {
  // input grad kernel is similar to forward kernel
  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int featureIdx = blockIdx.y;
  const int filterIdx = featureIdx / numFiltersInBlock;

  const int IOOffset =
      numFeatures * sequenceLength * batchIdx + featureIdx * sequenceLength;
  const scalar_t* inputFeature = &input[IOOffset];
  scalar_t* outputFeature = &output[IOOffset];
  const scalar_t* inputFilter = &filters[filterIdx * FS];

  assert(blockDim.x == SB);

  scalar_t filter[FS];

// The only change is loading the filter in reverse
#pragma unroll
  for (int i = 0; i < FS; ++i) {
    filter[i] = inputFilter[FS - i - 1];
  }

  __shared__ scalar_t temp[SB + FS];
  const int padding = FS - padding_l - 1;
  zeroSharedMem<FS, SB, padding>(temp);

  __syncthreads();

  const int numIterations = divUp<int, int>(sequenceLength, SB);

  for (int i = 0; i < numIterations; ++i) {
    // Read input into shared memory
    const int inputOffset = i * SB;

    load_input_to_shared<FS, SB, padding>(
        inputFeature,
        inputOffset,
        sequenceLength,
        i,
        numIterations,
        false,
        temp);

    __syncthreads();

    scalar_t out = 0;
#pragma unroll
    for (int j = 0; j < FS; ++j) {
      out += filter[j] * temp[tid + j];
    }

    // Write output
    const int outputOffset = inputOffset;
    if ((outputOffset + tid) < sequenceLength) {
      outputFeature[outputOffset + tid] = out;
    }

    __syncthreads();
  }
}

// This is by far the most expensive kernel in terms of time taken.
// Can be 16x slower than the forward or grad_wrt_input when filter size is 31
template <int FS, int SB, int padding_l, typename scalar_t>
__global__ void lightconv_grad_wrt_weights_firstpass_short_kernel(
    const scalar_t* input,
    const scalar_t* gradInput,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    int numHeads,
    float* output) {
  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int filterIdx = blockIdx.y;

  const int numIterations = divUp<int, int>(sequenceLength, SB);

  float* tempOutputGradWeight = &output[filterIdx * FS * minibatch];

  assert(blockDim.x == SB);

  __shared__ scalar_t tempInput[SB + FS];
  __shared__ scalar_t tempGradInput[SB + FS];

  // local weight accumulation
  float accumWeights[FS];

  // Initialize memory
  for (int i = 0; i < FS; ++i) {
    accumWeights[i] = float(0.0);
  }

  // loop over each sequence within filterblock
  for (int idxInFilterBlock = 0; idxInFilterBlock < numFiltersInBlock;
       ++idxInFilterBlock) {
    const int featureOffset = batchIdx * numFeatures * sequenceLength +
        (filterIdx * numFiltersInBlock + idxInFilterBlock) * sequenceLength;
    const scalar_t* inputFeature = &input[featureOffset];
    const scalar_t* gradInputFeature = &gradInput[featureOffset];

    zeroSharedMem<FS, SB, padding_l>(tempInput);
    zeroSharedMem<FS, SB, (FS / 2)>(tempGradInput);
    __syncthreads();

    for (int i = 0; i < numIterations; ++i) {
      const int inputOffset = i * SB;

      load_input_to_shared<FS, SB, padding_l>(
          inputFeature,
          inputOffset,
          sequenceLength,
          i,
          numIterations,
          false,
          tempInput);
      load_input_to_shared<FS, SB, (FS / 2)>(
          gradInputFeature,
          inputOffset,
          sequenceLength,
          i,
          numIterations,
          false,
          tempGradInput);

      __syncthreads();

      const int gradIndex = (FS / 2) + tid;
      scalar_t tempGrad = tempGradInput[gradIndex];

#pragma unroll
      for (int j = 0; j < FS; j++) {
        const int inputIndex = tid + j;
        accumWeights[j] += tempInput[inputIndex] * tempGrad;
      }

      __syncthreads();
    }
  }

  // Row-major sum
  for (int filterWeightIdx = 0; filterWeightIdx < FS; ++filterWeightIdx) {
    float temp;
    if (tid < sequenceLength) {
      temp = accumWeights[filterWeightIdx];
    } else {
      temp = float(0.0);
    }

    const int outputOffset = filterWeightIdx * minibatch + batchIdx;

    temp = blockReduce(temp);

    if (tid == 0) {
      tempOutputGradWeight[outputOffset] = temp;
    }
  }
}

template <int FS, int SB, typename scalar_t>
__global__ void lightconv_grad_wrt_weights_secondpass_short_kernel(
    const float* input,
    const int minibatch,
    const int numFiltersInBlock,
    scalar_t* output) {
  assert(blockDim.x == SB);

  const int tid = threadIdx.x;

  const int filterIdx = blockIdx.x;
  const int filterWeightIdx = blockIdx.y;

  const int inputOffset =
      filterIdx * FS * minibatch + filterWeightIdx * minibatch;
  const float* tempInput = &input[inputOffset];

  // read into shared memory for reduction
  int readIndex = tid;

  float sum = 0.0;
  while (readIndex < minibatch) {
    sum += tempInput[readIndex];
    readIndex += SB;
  }

  float temp = blockReduce(sum);

  if (tid == 0) {
    output[blockIdx.x * FS + blockIdx.y] = temp;
  }
}

// This is by far the most expensive kernel in terms of time taken.
// Can be 16x slower than the forward or grad_wrt_input when filter size is 31
template <int FS, int SB, int padding_l, typename scalar_t>
__global__ void lightconv_grad_wrt_weights_firstpass_kernel(
    const scalar_t* input,
    const scalar_t* gradInput,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    float* output) {
  assert(blockDim.x == SB);

  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int featureIdx = blockIdx.y;
  const int filterIdx = featureIdx / numFiltersInBlock;
  const int idxInFilterBlock = featureIdx % numFiltersInBlock;

  const int numIterations = divUp<int, int>(sequenceLength, SB);

  float temp;

  __shared__ scalar_t tempInput[SB + FS];
  __shared__ scalar_t tempGradInput[SB + FS];
  zeroSharedMem<FS, SB, padding_l>(tempInput);
  zeroSharedMem<FS, SB, (FS / 2)>(tempGradInput);
  __syncthreads();

  float accumWeights[FS];

  for (int i = 0; i < FS; ++i) {
    accumWeights[i] = float(0.0);
  }

  const int IOOffset =
      batchIdx * numFeatures * sequenceLength + featureIdx * sequenceLength;
  const scalar_t* inputFeature = &input[IOOffset];
  const scalar_t* gradInputFeature = &gradInput[IOOffset];
  float* tempOutputGradWeight =
      &output[filterIdx * FS * minibatch * numFiltersInBlock];

  for (int i = 0; i < numIterations; ++i) {
    const int inputOffset = i * SB;

    load_input_to_shared<FS, SB, padding_l>(
        inputFeature,
        inputOffset,
        sequenceLength,
        i,
        numIterations,
        false,
        tempInput);
    load_input_to_shared<FS, SB, (FS / 2)>(
        gradInputFeature,
        inputOffset,
        sequenceLength,
        i,
        numIterations,
        false,
        tempGradInput);
    __syncthreads();

#pragma unroll
    for (int j = 0; j < FS; ++j) {
      accumWeights[j] += tempInput[tid + j] * tempGradInput[tid + (FS / 2)];
    }

    __syncthreads();
  }

  // Row-major sum
  for (int filterWeightIdx = 0; filterWeightIdx < FS; ++filterWeightIdx) {
    // Write to shared memory before reduction
    if (tid < sequenceLength) {
      temp = accumWeights[filterWeightIdx];
    } else {
      temp = float(0.0);
    }

    temp = blockReduce(temp);

    const int outputOffset = filterWeightIdx * minibatch * numFiltersInBlock +
        batchIdx * numFiltersInBlock + idxInFilterBlock;

    if (tid == 0) {
      tempOutputGradWeight[outputOffset] = temp;
    }
  }
}

template <int FS, int SB, typename scalar_t>
__global__ void lightconv_grad_wrt_weights_secondpass_kernel(
    const float* input,
    const int minibatch,
    const int numFiltersInBlock,
    scalar_t* output) {
  assert(blockDim.x == SB);
  const int tid = threadIdx.x;

  // What is the id within a minibatch
  const int filterIdx = blockIdx.x;
  const int filterWeightIdx = blockIdx.y;

  const int inputOffset = filterIdx * FS * minibatch * numFiltersInBlock +
      filterWeightIdx * minibatch * numFiltersInBlock;
  const float* tempInput = &input[inputOffset];

  int readIndex = tid;

  float sum = float(0.0);
  while (readIndex < (minibatch * numFiltersInBlock)) {
    sum += tempInput[readIndex];
    readIndex += SB;
  }

  float temp = blockReduce(sum);

  if (tid == 0) {
    output[blockIdx.x * FS + blockIdx.y] = temp;
  }
}
