import os
from dataclasses import dataclass
import torch
import torch.utils.cpp_extension

cuda_source = """

#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>
#include <limits.h>
#include <cub/cub.cuh>
#include <iostream>

using namespace torch::indexing;

constexpr int kNumThreads = 1024;
constexpr float kNegInfinity = -std::numeric_limits<float>::infinity();
constexpr int kBlankIdx = 0;

__global__ void
falign_cuda_step_kernel(
  const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
    emissions_a,
  const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits>
    target_a,
  const int T, const int L, const int N, const int R, const int t, int start,
  int end, torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
             runningAlpha_a,
  torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits>
    backtrack_a, const bool normalize)
{
  int S = 2 * L + 1;
  
  int idx1 = (t % 2); // current time step frame for alpha
  int idx2 = ((t - 1) % 2); // previous time step frame for alpha
  
  // reset alpha and backtrack values
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
      runningAlpha_a[idx1][i] = kNegInfinity;
      backtrack_a[i] = -1;
  }
  // This could potentially be removed through careful indexing inside each thread
  // for the above for loop. But this is okay for now. 
  __syncthreads();

  if (t == 0) {
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
      int labelIdx = (i % 2 == 0) ? kBlankIdx : target_a[i / 2];
      runningAlpha_a[idx1][i] = emissions_a[0][labelIdx];
    }
    return;
  }

  using BlockReduce = cub::BlockReduce<float, kNumThreads>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ float maxValue;

  float threadMax;

  int startloop = start;

  threadMax = kNegInfinity;

  if (start == 0 && threadIdx.x == 0) {
    runningAlpha_a[idx1][0] =
      runningAlpha_a[idx2][0] + emissions_a[t][kBlankIdx];
    threadMax = max(threadMax, runningAlpha_a[idx1][0]);

    backtrack_a[0] = 0;
    // startloop += 1; // startloop is threadlocal meaning it would only be changed for threads entering this loop (ie threadIdx == 0)
  }
  if(start == 0) {
    startloop += 1;
  }

  for (int i = startloop + threadIdx.x; i < end; i += blockDim.x) {
    float x0 = runningAlpha_a[idx2][i];
    float x1 = runningAlpha_a[idx2][i - 1];
    float x2 = kNegInfinity;

    int labelIdx = (i % 2 == 0) ? kBlankIdx : target_a[i / 2];

    if (i % 2 != 0 && i != 1 && target_a[i / 2] != target_a[i / 2 - 1]) {
      x2 = runningAlpha_a[idx2][i - 2];
    }

    float result = 0.0;
    if (x2 > x1 && x2 > x0) {
      result = x2;
      backtrack_a[i] = 2;
    } else if (x1 > x0 && x1 > x2) {
      result = x1;
      backtrack_a[i] = 1;
    } else {
      result = x0;
      backtrack_a[i] = 0;
    }

    runningAlpha_a[idx1][i] = result + emissions_a[t][labelIdx];
    threadMax = max(threadMax, runningAlpha_a[idx1][i]);
  }

  float maxResult = BlockReduce(tempStorage).Reduce(threadMax, cub::Max());
  if (threadIdx.x == 0) {
    maxValue = maxResult;
  }

  __syncthreads();
  // normalize alpha values so that they don't overflow for large T
  if(normalize) {
      for (int i = threadIdx.x; i < S; i += blockDim.x) {
        runningAlpha_a[idx1][i] -= maxValue;
      }
  }
}

std::tuple<std::vector<int>, torch::Tensor, torch::Tensor>
falign_cuda(const torch::Tensor& emissions, const torch::Tensor& target, const bool normalize=false)
{
  TORCH_CHECK(emissions.is_cuda(), "need cuda tensors");
  TORCH_CHECK(target.is_cuda(), "need cuda tensors");
  TORCH_CHECK(target.device() == emissions.device(),
              "need tensors on same cuda device");
  TORCH_CHECK(emissions.dim() == 2 && target.dim() == 1, "invalid sizes");
  TORCH_CHECK(target.sizes()[0] > 0, "target size cannot be empty");



  int T = emissions.sizes()[0]; // num frames
  int N = emissions.sizes()[1]; // alphabet size
  int L = target.sizes()[0]; // label length
  const int S = 2 * L + 1;
  
  
  auto targetCpu = target.to(torch::kCPU);
  
  
  // backtrack stores the index offset fthe best path at current position  
  // We copy the values to CPU after running every time frame.
  
  auto backtrack = torch::zeros({ S }, torch::kInt32).to(emissions.device());
  auto backtrackCpu = torch::zeros(
    { T, S }, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  TORCH_CHECK(backtrack.is_cuda(), "need cuda tensors");
  TORCH_CHECK(!backtrackCpu.is_cuda(), "need cpu tensors");
  
 

  // we store only two time frames for alphas
  // alphas for compute current timeframe can be computed only from previous time frame.
  
  auto runningAlpha =
    torch::zeros(
      { 2, S },
      torch::TensorOptions().dtype(torch::kFloat).device(emissions.device()));
  auto alphaCpu =
    torch::zeros(
      { T, S },
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU));
  TORCH_CHECK(runningAlpha.is_cuda(), "need cuda tensors");
  TORCH_CHECK(!alphaCpu.is_cuda(), "need cpu tensors");

  auto stream = at::cuda::getCurrentCUDAStream();

  // CUDA accessors 
  auto emissions_a = emissions.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
  auto target_a = target.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>();
  auto runningAlpha_a =
    runningAlpha.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
  auto backtrack_a =
    backtrack.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>();

  
  // CPU accessors 
  auto targetCpu_a = targetCpu.accessor<int32_t, 1>();
  auto backtrackCpu_a = backtrackCpu.accessor<int32_t, 2>();
  auto aphaCpu_a = alphaCpu.accessor<float, 2>();
  
  // count the number of repeats in label
  int R = 0; 
  for (int i = 1; i < L; ++i) {
    if (targetCpu_a[i] == targetCpu_a[i - 1]) {
      ++R;
    }
  }
  TORCH_CHECK(T >= (L + R), "invalid sizes 2");


  int start = (T - (L + R)) > 0 ? 0 : 1;
  int end = (S == 1) ? 1 : 2;
  for (int t = 0; t < T; ++t) {
    if (t > 0) {
      if (T - t <= L + R) {
        if ((start % 2 == 1) &&
            (targetCpu_a[start / 2] != targetCpu_a[start / 2 + 1])) {
          start = start + 1;
        }
        start = start + 1;
      }
      if (t <= L + R) {
        if ((end % 2 == 0) && (end < 2 * L) &&
            (targetCpu_a[end / 2 - 1] != targetCpu_a[end / 2])) {
          end = end + 1;
        }
        end = end + 1;
      }
    }
    falign_cuda_step_kernel<<<1, kNumThreads, 0, stream>>>(
      emissions_a, target_a, T, L, N, R, t, start, end, runningAlpha_a,
      backtrack_a, normalize);

    backtrackCpu.index_put_({ t, Slice()}, backtrack.to(torch::kCPU));
    alphaCpu.index_put_({ t, Slice()}, runningAlpha.slice(0, t % 2, t % 2 + 1).to(torch::kCPU));
  }

  int idx1 = ((T - 1) % 2);
  int ltrIdx = runningAlpha[idx1][S - 1].item<float>() >
                   runningAlpha[idx1][S - 2].item<float>()
                 ? S - 1
                 : S - 2;

  std::vector<int> path(T);
  for (int t = T - 1; t >= 0; --t) {
    path[t] = (ltrIdx % 2 == 0) ? 0 : targetCpu_a[ltrIdx / 2];
    ltrIdx -= backtrackCpu_a[t][ltrIdx];
  }

  // returning runningAlpha, backtrackCpu for debugging purposes
  return std::make_tuple(path, alphaCpu, backtrackCpu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("falign", &falign_cuda, "falign cuda");
}
"""
falign_ext = torch.utils.cpp_extension.load_inline("falign", cpp_sources="", cuda_sources=cuda_source, extra_cflags=['-O3'], verbose=True )