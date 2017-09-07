/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

void TemporalConvolutionTBC_forward(
  THCudaTensor* input,
  THCudaTensor* output,
  THCudaTensor* weight,
  THCudaTensor* bias);

void TemporalConvolutionTBC_backward(
  THCudaTensor* _dOutput,
  THCudaTensor* _dInput,
  THCudaTensor* _dWeight,
  THCudaTensor* _dBias,
  THCudaTensor* _input,
  THCudaTensor* _weight);
