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
