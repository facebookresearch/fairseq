# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.modules.quantization.scalar import quantize_model_

def quantize_model_scalar(model, args):
    if getattr(args, 'scalar_quantization', False):
        # quantize_model edits the model in place
        quantize_model_(model, p=args.quant_noise, bits=8, update_step=1000)
    return model
