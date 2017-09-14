/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

void TemporalConvolutionTBC_forward(
  const char* dtype,
  void* input,
  void* output,
  void* weight,
  void* bias);

void TemporalConvolutionTBC_backward(
  const char* dtype,
  void* _dOutput,
  void* _dInput,
  void* _dWeight,
  void* _dBias,
  void* _input,
  void* _weight);
