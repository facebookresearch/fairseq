/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <string.h>
#include <stdexcept>
#include <ATen/ATen.h>


using at::Tensor;
extern THCState* state;

at::Type& getDataType(const char* dtype) {
  if (strcmp(dtype, "torch.cuda.FloatTensor") == 0) {
    return at::getType(at::kCUDA, at::kFloat);
  } else if (strcmp(dtype, "torch.FloatTensor") == 0) {
    return at::getType(at::kCPU, at::kFloat);
  } else {
    throw std::runtime_error(std::string("Unsupported data type: ") + dtype);
  }
}

inline at::Tensor t(at::Type& type, void* i) {
  return type.unsafeTensorFromTH(i, true);
}

void TemporalConvolutionTBC_forward(
  const char* dtype,
  void* _input,
  void* _output,
  void* _weight,
  void* _bias)
{
  auto& type = getDataType(dtype);
  Tensor input = t(type, _input);
  Tensor output = t(type, _output);
  Tensor weight = t(type, _weight);
  Tensor bias = t(type, _bias);

  auto input_size = input.sizes();
  auto output_size = output.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = output_size[2];
  auto olen = output_size[0];
  auto kw = weight.sizes()[0];
  int pad = (olen - ilen + kw - 1) / 2;

  // input * weights + bias -> output_features
  output.copy_(bias.expand(output.sizes()));
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // Note: gemm assumes column-major matrices
    // input    is l*m (row-major)
    // weight   is m*r (row-major)
    // output   is l*r (row-major)
    if (t > 0) {
      auto W = weight[k];
      auto I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      auto O = output.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      O.addmm_(I, W);
    }
  }
}

void TemporalConvolutionTBC_backward(
  const char* dtype,
  void* _dOutput,
  void* _dInput,
  void* _dWeight,
  void* _dBias,
  void* _input,
  void* _weight)
{
  auto& type = getDataType(dtype);
  Tensor dOutput = t(type, _dOutput);
  Tensor dInput = t(type, _dInput);
  Tensor dWeight = t(type, _dWeight);
  Tensor dBias = t(type, _dBias);
  Tensor input = t(type, _input);
  Tensor weight = t(type, _weight);

  auto input_size = input.sizes();
  auto output_size = dOutput.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = output_size[2];
  auto olen = output_size[0];
  auto kw = weight.sizes()[0];
  int pad = (olen - ilen + kw - 1) / 2;

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // dOutput * T(weight) -> dInput
    if (t > 0) {
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto dI = dInput.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      dI.addmm_(dO, weight[k].t());
    }
  }

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // T(input) * dOutput -> dWeight
    if (t > 0) {
      auto dW = dWeight[k];
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes}).t();
      dW.addmm_(I, dO);
    }
  }

  auto tmp = dOutput.sum(0, false);
  dBias.assign_(tmp.sum(0));
}
