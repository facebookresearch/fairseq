/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 */

#include "dynamicconv_cuda.cuh"
#include "dynamicconv_cuda_forward.cu"
#include "dynamicconv_cuda_backward.cu"
#include "../cuda_utils.cu"

// FS is filter size and kernels are specialized for filter sizes
template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void dynamicconv_forward_kernel(const scalar_t* input,
                                const scalar_t* weight,
                                int minibatch,
                                int sequenceLength,
                                int numFeatures,
                                int numFiltersInBlock,
                                int numHeads,
                                scalar_t* output) {
  assert(blockDim.x == SB);

  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int featureIdx = blockIdx.y;
  const int head = featureIdx / numFiltersInBlock;

  const int IOOffset = batchIdx * numFeatures * sequenceLength
                       + featureIdx * sequenceLength;
  const scalar_t* inputFeature = &input[IOOffset];
  scalar_t* outputFeature = &output[IOOffset];

  scalar_t filter[FS];

  __shared__ scalar_t tempInput[SB + FS];
  zeroSharedMem<FS, SB, padding_l>(tempInput);

  const int numIterations = divUp<int, int>(sequenceLength, SB);

  for (int i = 0; i < numIterations; ++i) {
    __syncthreads();
    const int inputOffset = i * SB;
    load_input_to_shared<FS, SB, padding_l>(inputFeature, inputOffset,
                                            sequenceLength, i,
                                            numIterations, false, tempInput);
    __syncthreads();
    if (inputOffset + tid < sequenceLength) {

      #pragma unroll
      for (int k = 0; k < FS; ++k) {
        const int filterOffset = batchIdx * numHeads * FS * sequenceLength
                                 + head * FS * sequenceLength
                                 + k * sequenceLength
                                 + i * SB + tid;
        filter[k] = weight[filterOffset];
      }

      scalar_t out = scalar_t(0.0);
      #pragma unroll
      for (int k = 0; k < FS; ++k) {
        out += filter[k] * tempInput[tid + k];
      }

      outputFeature[inputOffset + tid] = out;

    }
  }
}

template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void dynamicconv_backward_kernel(
    const scalar_t* gradOutput, // B * C * T
    const scalar_t* input, // B * C * T
    const scalar_t* weight,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    int numHeads,
    scalar_t* gradWeight,
    scalar_t* gradInput) { // B * H * k * T

  assert(blockDim.x == SB);

  // each block operates on a single batch and filter head
  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int headIdx = blockIdx.y;
  const int chunkIdx = blockIdx.z;

  const int numChunks = divUp<int, int>(sequenceLength, SB);
  const int inputOffset = chunkIdx * SB;

  // initialize shared memory for output gradient and input
  __shared__ scalar_t tempGradOutput[SB + FS];
  __shared__ scalar_t tempInput[SB + FS];
  const int padding = FS - padding_l - 1;

  zeroSharedMem<FS, SB, padding>(tempGradOutput);
  zeroSharedMem<FS, SB, padding_l>(tempInput);

  // initialize local filter and weight gradient sum arrays
  scalar_t tempGradSum[FS];
  scalar_t bfilter[FS];
  for (int k = 0; k < FS; ++k) {
    tempGradSum[k] = scalar_t(0.0);

    int idxOffset = inputOffset + tid + k - padding;
    if (idxOffset >= 0 && idxOffset < sequenceLength) {
      int bfilterOffset = batchIdx * numHeads * FS * sequenceLength
                          + headIdx * FS * sequenceLength
                          + (FS - k  - 1) * sequenceLength
                          + idxOffset;
      bfilter[k] = weight[bfilterOffset];
    } else {
      bfilter[k] = scalar_t(0.0);
    }
  }


  // iterate over filter block
  for (int featureIdx = 0; featureIdx < numFiltersInBlock; ++featureIdx) {
    __syncthreads();

    // load input and output gradient for this channel and chunk
    const int IOOffset = batchIdx * numFeatures * sequenceLength
                         + (headIdx * numFiltersInBlock + featureIdx) * sequenceLength;
    const scalar_t* inputFeature = &input[IOOffset];
    const scalar_t* gradOutputFeature = &gradOutput[IOOffset];
    scalar_t* gradInputFeature = &gradInput[IOOffset];

    load_input_to_shared<FS, SB, padding>(gradOutputFeature, inputOffset,
                                            sequenceLength, chunkIdx,
                                            numChunks, true, tempGradOutput);
    load_input_to_shared<FS, SB, padding_l>(inputFeature, inputOffset,
                                            sequenceLength, chunkIdx,
                                            numChunks, true, tempInput);
    __syncthreads();
 
    // sum input and weight gradients
    scalar_t out = scalar_t(0.0);
    #pragma unroll
    for (int k = 0; k < FS; ++k) {
      tempGradSum[k] += tempInput[tid + k] * tempGradOutput[tid + padding];
      out += bfilter[k] * tempGradOutput[tid + k];
    }
    
    if (inputOffset + tid < sequenceLength) {
      gradInputFeature[inputOffset + tid] = out;
    }
  }

  const int gradOffset = batchIdx * numHeads * FS * sequenceLength
               + headIdx * FS * sequenceLength;
  scalar_t *gradWeightFeature = &gradWeight[gradOffset];

  // write weight gradient
  if (inputOffset + tid < sequenceLength) {
    for (int k = 0; k < FS; ++k) {
      const int outputOffset = k * sequenceLength + inputOffset + tid;
      gradWeightFeature[outputOffset] = tempGradSum[k];
    }
  }
}
