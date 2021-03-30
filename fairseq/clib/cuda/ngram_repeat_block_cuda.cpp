/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

#include <torch/extension.h>
#include <vector>

/*
CPP Binding for CUDA OP
*/

// CUDA forward declarations
torch::Tensor ngram_repeat_block_cuda_forward(torch::Tensor tokens,
                                              torch::Tensor lprobs, int bsz,
                                              int step, int beam_size,
                                              int no_repeat_ngram_size);

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// Input check and call to CUDA OP
// Backward method not required
torch::Tensor ngram_repeat_block_forward(torch::Tensor tokens,
                                         torch::Tensor lprobs, int bsz,
                                         int step, int beam_size,
                                         int no_repeat_ngram_size) {
  CHECK_INPUT(tokens);
  CHECK_INPUT(lprobs);
  assert(bsz > 0);
  assert(step >= 0);
  assert(beam_size > 0);
  assert(no_repeat_ngram_size > 0);

  return ngram_repeat_block_cuda_forward(tokens, lprobs, bsz, step, beam_size,
                                         no_repeat_ngram_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ngram_repeat_block_forward,
        "No Repeat Ngram Block forward (CUDA)");
}
