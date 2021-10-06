/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h> // @manual=//caffe2/aten:ATen-cu
#include <cuda_runtime.h>
#include <algorithm> // std::min/max
#include <cub/cub.cuh>

#include "alignment_train_cuda.h"
#include "utils.h"

namespace {

// The thread block length in threads along the X dimension
constexpr int BLOCK_DIM_X = 128;
// The thread block length in threads along the Y dimension
constexpr int BLOCK_DIM_Y = 8;
// The thread block length in threads for scan operation
constexpr int SCAN_BLOCK = 512;

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(
        stderr,
        "\nGPUassert: %s %s %d\n",
        cudaGetErrorString(code),
        file,
        line);
    if (abort)
      exit(code);
  }
}

template <typename T>
struct Prod {
  /// prod operator, returns <tt>a * b</tt>
  __host__ __device__ __forceinline__ T
  operator()(const T& a, const T& b) const {
    return a * b;
  }
};

template <typename T>
struct BlockPrefixProdCallbackOp {
  // Running prefix
  T running_total;

  // Constructor
  __device__ BlockPrefixProdCallbackOp(T running_total)
      : running_total(running_total) {}

  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ T operator()(const T block_aggregate) {
    T old_prefix = running_total;
    running_total *= block_aggregate;
    return old_prefix;
  }
};

template <typename T>
struct BlockPrefixSumCallbackOp {
  // Running prefix
  T running_total;

  // Constructor
  __device__ BlockPrefixSumCallbackOp(T running_total)
      : running_total(running_total) {}

  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ T operator()(const T block_aggregate) {
    T old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <typename T>
__global__ void oneMinusPKernel(
    const T* __restrict__ p_choose,
    T* __restrict__ cumprod_1mp,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len) {
  for (uint32_t b = blockIdx.x; b < bsz; b += gridDim.x) {
    for (uint32_t tgt = threadIdx.y; tgt < tgt_len; tgt += blockDim.y) {
      for (uint32_t src = threadIdx.x; src < src_len; src += blockDim.x) {
        uint32_t idx = b * tgt_len * src_len + tgt * src_len + src;
        cumprod_1mp[idx] = 1 - p_choose[idx];
      }
    }
  }
}

template <typename T, int TPB>
__global__ void innermostScanKernel(
    T* __restrict__ cumprod_1mp,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len) {
  for (uint32_t b = blockIdx.y; b < bsz; b += gridDim.y) {
    for (uint32_t tgt = blockIdx.x; tgt < tgt_len; tgt += gridDim.x) {
      // Specialize BlockScan for a 1D block of TPB threads on type T
      typedef cub::BlockScan<T, TPB> BlockScan;
      // Allocate shared memory for BlockScan
      __shared__ typename BlockScan::TempStorage temp_storage;
      // Initialize running total
      BlockPrefixProdCallbackOp<T> prefix_op(1);

      const uint32_t tid = threadIdx.x;
      for (uint32_t block_src = 0; block_src < src_len;
           block_src += blockDim.x) {
        uint32_t src = block_src + tid;
        uint32_t idx = b * tgt_len * src_len + tgt * src_len + src;
        T thread_data = (src < src_len) ? cumprod_1mp[idx] : (T)0;

        // Collectively compute the block-wide inclusive prefix sum
        BlockScan(temp_storage)
            .ExclusiveScan(thread_data, thread_data, Prod<T>(), prefix_op);
        __syncthreads();

        // write the scanned value to output
        if (src < src_len) {
          cumprod_1mp[idx] = thread_data;
        }
      }
    }
  }
}

template <typename T>
__global__ void clampKernel(
    const T* __restrict__ cumprod_1mp,
    T* __restrict__ cumprod_1mp_clamp,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len,
    T min_val,
    T max_val) {
  for (uint32_t b = blockIdx.x; b < bsz; b += gridDim.x) {
    for (uint32_t tgt = threadIdx.y; tgt < tgt_len; tgt += blockDim.y) {
      for (uint32_t src = threadIdx.x; src < src_len; src += blockDim.x) {
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
__global__ void initAlphaCUDAKernel(
    T* alpha,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len) {
  // alpha[:, 0, 0] = 1.0
  for (uint32_t b = blockIdx.x; b < bsz; b += gridDim.x) {
    alpha[b * tgt_len * src_len] = (T)1.0;
  }
}

template <typename T, int TPB>
__global__ void alignmentTrainCUDAKernel(
    const T* __restrict__ p_choose,
    const T* __restrict__ cumprod_1mp,
    const T* __restrict__ cumprod_1mp_clamp,
    T* __restrict__ alpha,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len,
    uint32_t tgt) {
  for (uint32_t b = blockIdx.x; b < bsz; b += gridDim.x) {
    // Specialize BlockScan for a 1D block of TPB threads on type T
    typedef cub::BlockScan<T, TPB> BlockScan;

    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;
    // Initialize running total
    BlockPrefixSumCallbackOp<T> prefix_op(0);

    uint32_t b_offset = b * tgt_len * src_len;
    const uint32_t tid = threadIdx.x;
    for (uint32_t block_src = 0; block_src < src_len; block_src += blockDim.x) {
      uint32_t src = block_src + tid;
      // Obtain a segment of consecutive items that are blocked across threads
      uint32_t inout_idx, alpha_idx;
      if (tgt == 0) {
        // both alpha and other input index is [b][0][src]
        alpha_idx = b_offset + src;
      } else {
        // alpha index is [b][tgt-1][src]
        alpha_idx = b_offset + (tgt - 1) * src_len + src;
      }
      inout_idx = b_offset + tgt * src_len + src;
      T thread_data = (T)0;
      if (src < src_len) {
        thread_data = alpha[alpha_idx] / cumprod_1mp_clamp[inout_idx];
      }

      // Collectively compute the block-wide inclusive prefix sum
      BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, prefix_op);
      __syncthreads();

      if (src < src_len) {
        T out = thread_data * p_choose[inout_idx] * cumprod_1mp[inout_idx];
        // Clamps all elements into the range [ 0, 1.0 ]
        alpha[inout_idx] = std::min<T>(std::max<T>(out, 0), (T)1.0);
      }
    }
  }
}

template <typename T>
void exclusiveCumprod(
    const T* p_choose,
    T* cumprod_1mp,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len,
    uint32_t max_grid_x,
    uint32_t max_grid_y,
    cudaStream_t& stream) {
  // cumprod_1mp = 1 - p_choose
  dim3 grid(std::min<T>(max_grid_x, bsz), 1, 1);
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
  oneMinusPKernel<T><<<grid, block, 0, stream>>>(
      p_choose, cumprod_1mp, bsz, tgt_len, src_len);
  gpuErrchk(cudaGetLastError());

  // scan on the innermost dimension of cumprod_1mp
  // cumprod_1mp = cumprod(cumprod_1mp)
  dim3 grid_scan(
      std::min<T>(max_grid_x, tgt_len), std::min<T>(max_grid_y, bsz), 1);
  innermostScanKernel<T, SCAN_BLOCK><<<grid_scan, SCAN_BLOCK, 0, stream>>>(
      cumprod_1mp, bsz, tgt_len, src_len);
  gpuErrchk(cudaGetLastError());
}

template <typename T>
void alignmentTrainCUDAImpl(
    const T* p_choose,
    T* alpha,
    uint32_t bsz,
    uint32_t tgt_len,
    uint32_t src_len,
    float eps) {
  // p_choose: bsz , tgt_len, src_len
  // cumprod_1mp: bsz , tgt_len, src_len
  // cumprod_1mp_clamp : bsz, tgt_len, src_len
  // alpha: bsz, tgt_len, src_len
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  uint32_t max_grid_x = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  uint32_t max_grid_y = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];

  // Implementing exclusive cumprod.
  // cumprod_1mp = cumprod(1 - p_choose)
  // There is cumprod in pytorch, however there is no exclusive mode.
  // cumprod(x) = [x1, x1x2, x2x3x4, ..., prod_{i=1}^n x_i]
  // exclusive means
  // cumprod(x) = [1, x1, x1x2, x1x2x3, ..., prod_{i=1}^{n-1} x_i]
  uint32_t elements = bsz * tgt_len * src_len;
  T* cumprod_1mp;
  gpuErrchk(cudaMalloc(&cumprod_1mp, elements * sizeof(T)));
  exclusiveCumprod<T>(
      p_choose,
      cumprod_1mp,
      bsz,
      tgt_len,
      src_len,
      max_grid_x,
      max_grid_y,
      stream);

  // clamp cumprod_1mp to the range [eps, 1.0]
  T* cumprod_1mp_clamp;
  gpuErrchk(cudaMalloc(&cumprod_1mp_clamp, elements * sizeof(T)));
  dim3 grid_clamp(std::min<T>(max_grid_x, bsz), 1, 1);
  dim3 block_clamp(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
  clampKernel<T><<<grid_clamp, block_clamp, 0, stream>>>(
      cumprod_1mp, cumprod_1mp_clamp, bsz, tgt_len, src_len, (T)eps, (T)1.0);
  gpuErrchk(cudaGetLastError());

  // ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))
  dim3 grid_init(std::min<int>(max_grid_x, bsz), 1, 1);
  initAlphaCUDAKernel<T>
      <<<grid_init, 1, 0, stream>>>(alpha, bsz, tgt_len, src_len);
  gpuErrchk(cudaGetLastError());

  const int grid = std::min(bsz, max_grid_x);

  for (uint32_t i = 0; i < tgt_len; i++) {
    alignmentTrainCUDAKernel<T, SCAN_BLOCK><<<grid, SCAN_BLOCK, 0, stream>>>(
        p_choose,
        cumprod_1mp,
        cumprod_1mp_clamp,
        alpha,
        bsz,
        tgt_len,
        src_len,
        i);
    gpuErrchk(cudaGetLastError());
  }

  gpuErrchk(cudaFree(cumprod_1mp));
  gpuErrchk(cudaFree(cumprod_1mp_clamp));
}

} // namespace

void alignmentTrainCUDAWrapper(
    const torch::Tensor& p_choose,
    torch::Tensor& alpha,
    float eps) {
  // p_choose dimension: bsz, tgt_len, src_len
  uint32_t bsz = p_choose.size(0);
  uint32_t tgt_len = p_choose.size(1);
  uint32_t src_len = p_choose.size(2);

  cudaSetDevice(p_choose.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::ScalarType::Half,
      torch::ScalarType::BFloat16,
      p_choose.scalar_type(),
      "alignmentTrainCUDAImpl",
      [&]() {
        alignmentTrainCUDAImpl<scalar_t>(
            p_choose.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            bsz,
            tgt_len,
            src_len,
            eps);
      });
}
