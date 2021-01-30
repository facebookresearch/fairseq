/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

/*
Kernel implementation for blocking repeated n-grams.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>
#include <vector>

// Ban repeated ngrams of length = 'no_repeat_ngram_size'
__global__ void banRepeatedTokens(long* __restrict__ tokens,
                                  float* __restrict__ lprobs,
                                  int max_predict_len, int vocab_size,
                                  int no_repeat_ngram_size) {
  auto row = blockIdx.x;
  auto col = threadIdx.x;
  auto start = row * (max_predict_len) + col;
  // Each thread compares ngram starting from
  // thread index with final ngram starting from
  // step - no_repeat_ngram_size +2
  auto check_start_pos = blockDim.x;
  auto lprob_start = row * vocab_size;
  bool is_banned = true;
  extern __shared__ long tokens_shm[];
  tokens_shm[col] = tokens[start];
  if (col == blockDim.x - 1) {
     for (int i=1; i<no_repeat_ngram_size; i++){
	if (col+i < max_predict_len){
          tokens_shm[col + i] = tokens[start + i];
	}
    }
  }
  __syncthreads();

  for (int k = 0; k < no_repeat_ngram_size - 1; k++) {
    if (tokens_shm[col + k] != tokens_shm[check_start_pos + k]) {
      is_banned = false;
    }
  }
  if (is_banned == true) {
    auto token_to_be_banned = tokens_shm[col + no_repeat_ngram_size - 1];
    lprobs[lprob_start + token_to_be_banned] = -INFINITY;
  }
}

// Allocate blocks and threads based on
// batch size and sequence length and launch
// kernel
torch::Tensor ngram_repeat_block_cuda_forward(const torch::Tensor tokens,
                                              torch::Tensor lprobs, int bsz,
                                              int step, int beam_size,
                                              int no_repeat_ngram_size) {
  int threads = step - no_repeat_ngram_size + 2;
  if (threads <= 0) return lprobs;
  int max_predict_len = tokens.size(1);
  int vocab_size = lprobs.size(1);
  auto token_ptr = tokens.data_ptr<long>();
  auto lprob_ptr = lprobs.data_ptr<float>();
  int blocks = bsz * beam_size;
  int shared_mem_size = (step + 1) * sizeof(long);

  // Launching N blocks where N is number of samples in a batch (beams*bsz)
  // Launching T threads where T is number of previous ngrams in a sample
  // Allocating shared mem per block for fastser access of input tokens since
  // each token will be accessed N times to compare with current Ngram where
  // N is Ngram size.
  banRepeatedTokens<<<blocks, threads, shared_mem_size>>>(
      token_ptr, lprob_ptr, max_predict_len, vocab_size, no_repeat_ngram_size);
  return lprobs;
}
