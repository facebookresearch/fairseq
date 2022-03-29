/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "alignment_train_cuda.h"
#include "utils.h"

namespace {

void alignmentTrainCUDA(
    const torch::Tensor& p_choose,
    torch::Tensor& alpha,
    float eps) {
  CHECK_INPUT(p_choose);
  CHECK_INPUT(alpha);

  alignmentTrainCUDAWrapper(p_choose, alpha, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "alignment_train_cuda",
      &alignmentTrainCUDA,
      "expected_alignment_from_p_choose (CUDA)");
}

} // namespace
