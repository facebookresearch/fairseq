#include <stdio.h>
#include <THC/THC.h>
#include <ATen/ATen.h>


using at::Tensor;
extern THCState* state;


inline at::Tensor t(THCudaTensor* i) {
  return at::getType(at::kCUDA, at::kFloat).unsafeTensorFromTH(i, true);
}

extern "C" void TemporalConvolutionTBC_forward(
  THCudaTensor* _input,
  THCudaTensor* _output,
  THCudaTensor* _weight,
  THCudaTensor* _bias)
{
  Tensor input = t(_input);
  Tensor output = t(_output);
  Tensor weight = t(_weight);
  Tensor bias = t(_bias);

  auto W = weight.data<float>();
  auto I = input.data<float>();
  auto O = output.data<float>();

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
    if (t > 0)
      THCudaBlas_Sgemm(
          state,
          'n',
          'n',
          outputPlanes, // r
          batchSize * t, // l
          inputPlanes, // m
          1, // alpha
          W + k * weight.strides()[0],
          outputPlanes, // r
          I + iShift * input.strides()[0],
          input.strides()[1], // >=m
          1, // beta
          O + oShift * output.strides()[0],
          output.strides()[1] // r
          );
  }
}

extern "C" void TemporalConvolutionTBC_backward(
  THCudaTensor* _dOutput,
  THCudaTensor* _dInput,
  THCudaTensor* _dWeight,
  THCudaTensor* _dBias,
  THCudaTensor* _input,
  THCudaTensor* _weight)
{
  Tensor dOutput = t(_dOutput);
  Tensor dInput = t(_dInput);
  Tensor dWeight = t(_dWeight);
  Tensor dBias = t(_dBias);
  Tensor input = t(_input);
  Tensor weight = t(_weight);

  auto dO = dOutput.data<float>();
  auto dI = dInput.data<float>();
  auto dW = dWeight.data<float>();
  auto I = input.data<float>();
  auto W = weight.data<float>();

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
    // Note: gemm assumes column-major matrices
    // dOutput is l*m (row-major)
    // weight  is r*m (row-major)
    // dInput  is l*r (row-major)
    if (t > 0)
      THCudaBlas_Sgemm(
          state,
          't',
          'n',
          inputPlanes, // r
          batchSize * t, // l
          outputPlanes, // m
          1, // alpha
          W + k * weight.strides()[0],
          outputPlanes, // m
          dO + oShift * dOutput.strides()[0],
          dOutput.strides()[1], // m
          1, // beta
          dI + iShift * dInput.strides()[0],
          dInput.strides()[1] // m
          );
  }

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // Note: gemm assumes column-major matrices
    // Input    is m*l (row-major)
    // dOutput  is m*r (row-major)
    // dWeight  is l*r (row-major)
    if (t > 0)
      THCudaBlas_Sgemm(
          state,
          'n',
          't',
          outputPlanes, // r
          inputPlanes, // l
          batchSize * t, // m
          1, //scale, // alpha
          dO + oShift * dOutput.strides()[0],
          dOutput.strides()[1], // r
          I + iShift * input.strides()[0],
          input.strides()[1], // l
          1, // beta
          dW + k * dWeight.strides()[0],
          outputPlanes // r
          );
   }

   auto tmp = dOutput.sum(0, false);
   at::sum_out(tmp, 0, dBias);
}
