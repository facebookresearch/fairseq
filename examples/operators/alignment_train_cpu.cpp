/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h> // @manual=//caffe2:torch_extension
#include <algorithm>

namespace {

template <typename T>
void exclusiveCumprod(
    const T* p_choose,
    T* cumprod_1mp,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len) {
  // cumprod_1mp = 1 - p_choose
  for (uint32_t b = 0; b < bsz; b++) {
    for (uint32_t tgt = 0; tgt < tgt_len; tgt++) {
      for (uint32_t src = 0; src < src_len; src++) {
        uint32_t idx = b * tgt_len * src_len + tgt * src_len + src;
        cumprod_1mp[idx] = 1 - p_choose[idx];
      }
    }
  }

  // Implementing exclusive cumprod in the innermost dimension
  // cumprod_1mp = cumprod(1 - p_choose)
  // There is cumprod in pytorch, however there is no exclusive mode.
  // cumprod(x) = [x1, x1x2, x2x3x4, ..., prod_{i=1}^n x_i]
  // exclusive means
  // cumprod(x) = [1, x1, x1x2, x1x2x3, ..., prod_{i=1}^{n-1} x_i]
  for (uint32_t b = 0; b < bsz; b++) {
    for (uint32_t tgt = 0; tgt < tgt_len; tgt++) {
      uint32_t idx_offset = b * tgt_len * src_len + tgt * src_len;
      T prev = cumprod_1mp[idx_offset];
      // index [b][tgt][0]
      cumprod_1mp[idx_offset] = (T)1.0;
      T curr;
      for (uint32_t src = 1; src < src_len; src++) {
        uint32_t idx = idx_offset + src;
        curr = cumprod_1mp[idx];
        cumprod_1mp[idx] = cumprod_1mp[idx - 1] * prev;
        prev = curr;
      }
    }
  }
}

template <typename T>
void clamp(
    const T* cumprod_1mp,
    T* cumprod_1mp_clamp,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len,
    T min_val,
    T max_val) {
  for (uint32_t b = 0; b < bsz; b++) {
    for (uint32_t tgt = 0; tgt < tgt_len; tgt++) {
      for (uint32_t src = 0; src < src_len; src++) {
        uint32_t idx = b * tgt_len * src_len + tgt * src_len + src;
        if (cumprod_1mp[idx] < min_val) {
          cumprod_1mp_clamp[idx] = min_val;
        } else if (cumprod_1mp[idx] > max_val) {
          cumprod_1mp_clamp[idx] = max_val;
        } else {
          cumprod_1mp_clamp[idx] = cumprod_1mp[idx];
        }
      }
    }
  }
}

template <typename T>
void alignmentTrainCPUImpl(
    const T* p_choose,
    T* alpha,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len,
    float eps) {
  // p_choose: bsz , tgt_len, src_len
  // cumprod_1mp: bsz , tgt_len, src_len
  // cumprod_1mp_clamp : bsz, tgt_len, src_len
  // alpha: bsz + 1, tgt_len, src_len

  uint32_t elements = bsz * tgt_len * src_len;
  T* cumprod_1mp = new T[elements];
  T* cumprod_1mp_clamp = new T[elements];

  exclusiveCumprod<T>(p_choose, cumprod_1mp, bsz, tgt_len, src_len);
  clamp<T>(
      cumprod_1mp, cumprod_1mp_clamp, bsz, tgt_len, src_len, (T)eps, (T)1.0);

  // ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))

  // Initialize alpha [:, 0, 0]
  for (uint32_t b = 0; b < bsz; b++) {
    alpha[b * tgt_len * src_len] = 1.0;
  }

  for (uint32_t tgt = 0; tgt < tgt_len; tgt++) {
    for (uint32_t b = 0; b < bsz; b++) {
      uint32_t alpha_idx, inout_idx;
      T prev_scan = 0, curr_scan, out;
      for (uint32_t src = 0; src < src_len; src++) {
        // Apply scan/cumsum
        if (tgt == 0) {
          // alpha index is [b][tgt][src]
          alpha_idx = b * tgt_len * src_len + src;
        } else {
          // alpha index is [b][tgt-1][src]
          alpha_idx = b * tgt_len * src_len + (tgt - 1) * src_len + src;
        }
        // input index is [b][tgt][src]
        inout_idx = b * tgt_len * src_len + tgt * src_len + src;
        curr_scan = prev_scan + alpha[alpha_idx] / cumprod_1mp_clamp[inout_idx];

        out = curr_scan * p_choose[inout_idx] * cumprod_1mp[inout_idx];
        alpha[inout_idx] = std::min<T>(std::max<T>(out, 0), 1.0);
        prev_scan = curr_scan;
      }
    }
  }

  free(cumprod_1mp);
  free(cumprod_1mp_clamp);
}

void alignmentTrainCPU(
    const torch::Tensor& p_choose,
    torch::Tensor& alpha,
    float eps) {
  uint32_t bsz = p_choose.size(0);
  uint32_t tgt_len = p_choose.size(1);
  uint32_t src_len = p_choose.size(2);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::ScalarType::Half,
      torch::ScalarType::BFloat16,
      p_choose.scalar_type(),
      "alignmentCPUImpl",
      [&]() {
        alignmentTrainCPUImpl<scalar_t>(
            p_choose.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            bsz,
            tgt_len,
            src_len,
            eps);
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "alignment_train_cpu",
      &alignmentTrainCPU,
      "expected_alignment_from_p_choose (CPU)");
}

} // namespace
