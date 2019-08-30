/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * 
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


template <typename U, typename V>	
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {	
  return (a + b - 1) / b;	
}


template<int FS, int SB, int padding_l, typename scalar_t>
__inline__ __device__
void zeroSharedMem(scalar_t* data) {
  /*
    Given an array of length FS + SB, zero out the first padding_l and last
    (FS - padding_l) values in the array
  */

  int tid = threadIdx.x;

  if (FS < SB) {

    // zero all if we have enough threads in a block to do all of them
    if (tid < padding_l || tid > SB - FS + padding_l - 1) {
      data[tid] = scalar_t(0.0);
    }
  } else {

    // otherwise zero out one block at a time
    const int numIterations = divUp<int, int>(FS, SB);
    for (int i = 0; i < numIterations; i++) {
      int offset = i * SB;
      if (tid + offset < padding_l) {
        data[tid + offset] = scalar_t(0.0);
      } else if (tid + offset < FS) {
        data[SB + tid + offset] = scalar_t(0.0);
      }
    }
  }
}

template<typename scalar_t>
__inline__ __device__
scalar_t warpReduce(scalar_t data) {
  /*
    Reduce an array within each warp. After processing all values in warp will
    caontain the sum of all original values in that warp.

    data - pointer to data to reduce
  */
  data += __shfl_xor_sync(SHFL_MASK, data, 16);
  data += __shfl_xor_sync(SHFL_MASK, data, 8);
  data += __shfl_xor_sync(SHFL_MASK, data, 4);
  data += __shfl_xor_sync(SHFL_MASK, data, 2);
  data += __shfl_xor_sync(SHFL_MASK, data, 1);
  return data;
}

template<typename scalar_t>
__inline__ __device__
scalar_t blockReduce(scalar_t data) {
  /*
     Reduce an entire array on the block level. After processing, the
     first value in the array will contain the reduced sum.

     data - pointer to data to reduce
  */

  static __shared__ scalar_t warpSum[32];
  const int tid = threadIdx.x;
  int wid = tid / 32;
  int lane = tid % 32;

  __syncthreads();

  // reduce each warp then write to shared memory
  scalar_t sum = warpReduce(data);
  if (lane == 0) {
    warpSum[wid] = sum;
  }
  
  __syncthreads();

  scalar_t v;
  // perform final sum of partial warp sums
  if (tid < blockDim.x / 32) {
    v = warpSum[lane];
  } else {
    v = scalar_t(0.0);
  }

  if (wid == 0) {
    v = warpReduce(v);
  }
  __syncthreads();

  return v;
}

void checkCudaStatus(cudaError_t status, int lineNumber = -1) {

  if (status != cudaSuccess) {
    std::cout << cudaGetErrorString(status)
              << " at line " << lineNumber << std::endl;
    std::cout << "Exiting" << std::endl;
    exit(1);
  }
}

template<int FS, int SB, int padding_l, typename scalar_t>
__device__
void load_input_to_shared(const scalar_t* input, // global memory
                          int inputOffset, int sequenceLength,
                          int iteration, int numIterations,
                          bool no_prev, scalar_t* output /* shared memory */) {
  /*
    Load a block size of input into shared memory with
    right and left overhang of total size FS. If previously
    loaded memory, overlap will be shifted over to reduce
    global memory access

    input - pointer to start of channel sequence
    inputOffset - how far in the sequence to start loading
    sequenceLength - total length of sequence
    iteration - which block of sequence we are loading
    numIterations - total number of blocks to load
    no_prev - whether to load the whole block if the previous block
              wasn't loaded
    output - shared memory to write input to
  */

  const int tid = threadIdx.x;

  // Load the left "overhang" of input
  if (iteration > 0) {
    if (padding_l < SB) {

      // load all at once
      if (tid < padding_l) {
        output[tid] = (no_prev) ? input[inputOffset - padding_l + tid] : output[tid + SB];
      }
    } else {

      // load in chunks of size SB
      int numIterations = divUp<int, int>(padding_l, SB);
      for (int i = 0; i < numIterations; i++) {
        int offset = i * SB;
        if ((tid + offset) < padding_l) {
          output[tid + offset] = (no_prev) ? input[inputOffset - padding_l + tid + offset] : output[tid + offset + SB];
        }
      }
    }
  }

  // Load the right "overhang" of input
  if (iteration < (numIterations - 1)) {
    const int elementsLeft = sequenceLength - (iteration+1) * SB;

    if ((FS - padding_l) < SB) {

      // load all at once
      if (tid < (FS - padding_l)) {
          output[padding_l + SB + tid] = (tid < elementsLeft) ? input[inputOffset + SB + tid] : scalar_t(0.0);
      }
    } else {

      // load in chunks of size SB
      int numIterations = divUp<int, int>(FS - padding_l, SB);
      for (int i = 0; i < numIterations; i++) {
        int offset = i * SB;
        if ((tid + offset) < (FS - padding_l)) {
          output[padding_l + SB + tid + offset] = ((tid + offset) < elementsLeft) ? input[inputOffset + SB + tid + offset] : scalar_t(0.0);
        }
      }
    }
  }

  // We should also clear out the right "overhang"
  if (iteration == (numIterations - 1)) {
    if ((FS - padding_l) < SB) {

      // clear out all at once
      if (tid < (FS - padding_l)) {
          output[padding_l + SB + tid] = scalar_t(0.0);
      }
    } else {

      // clear in chunks of size SB
      int numIterations = divUp<int, int>(FS - padding_l, SB);
      for (int i = 0; i < numIterations; i++) {
        int offset = i * SB;
        if ((tid + offset) < (FS - padding_l)) {
          output[padding_l + SB + tid + offset] = scalar_t(0.0);
        }
      }
    }
  }
  output[tid + padding_l] = ((inputOffset + tid) < sequenceLength) ? input[inputOffset + tid] : scalar_t(0.0);
}
