# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def gen_forward():

    kernels = [3, 5, 7, 15, 31, 63, 127, 255]
    blocks = [32, 64, 128, 256]

    head = """
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dynamicconv_cuda.cuh"

std::vector<at::Tensor> dynamicconv_cuda_forward(at::Tensor input, at::Tensor weight, int padding_l) {

    at::DeviceGuard g(input.device());
    const auto minibatch = input.size(0);
    const auto numFeatures = input.size(1);
    const auto sequenceLength = input.size(2);

    const auto numHeads = weight.size(1);
    const auto filterSize = weight.size(2);

    const auto numFiltersInBlock = numFeatures / numHeads;
    const dim3 blocks(minibatch, numFeatures);

    auto output = at::zeros_like(input);
    auto stream = at::cuda::getCurrentCUDAStream();
"""

    switch = """
    switch(filterSize) {
"""

    case_k = """
        case {k}:
"""

    main_block = """
            if (padding_l == {pad}) {{
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {{
                    dynamicconv_forward_kernel<{k}, {b_size}, {pad}, scalar_t>
                    <<<blocks, {b_size}, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }}));
            }} else
"""

    bad_padding = """
            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;\n
"""

    end = """
        default:
            std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
    }

    return {output};
}
"""

    with open("dynamicconv_cuda_forward.cu", "w") as forward:
        forward.write(head)
        forward.write(switch)
        for k in kernels:
            b_size = 32
            for b in blocks:
                if b > k:
                    b_size = b
                    break
            forward.write(case_k.format(k=k))
            for pad in [k // 2, k - 1]:
                forward.write(main_block.format(k=k, b_size=b_size, pad=pad))
            forward.write(bad_padding)
        forward.write(end)


def gen_backward():

    kernels = [3, 5, 7, 15, 31, 63, 127, 255]
    thresh = [512, 512, 512, 512, 512, 380, 256, 256]
    min_block = [64, 64, 64, 64, 64, 64, 128, 256]
    seqs = [32 * x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]

    head = """
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dynamicconv_cuda.cuh"

std::vector<at::Tensor> dynamicconv_cuda_backward(at::Tensor gradOutput, int padding_l, at::Tensor input, at::Tensor weight) {

    at::DeviceGuard g(input.device());
    const auto minibatch = input.size(0);
    const auto numFeatures = input.size(1);
    const auto sequenceLength = input.size(2);

    const auto numHeads = weight.size(1);
    const auto filterSize = weight.size(2);

    const auto numFiltersInBlock = numFeatures / numHeads;
    auto numChunks = 1;

    auto gradInput = at::zeros_like(input);
    auto gradWeight = at::zeros_like(weight);
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks(minibatch, numHeads, numChunks);
"""

    sequence_if = """
    if (sequenceLength < {seq}) {{
        switch(filterSize) {{
"""

    case_k = """
            case {k}:
"""

    chunks_reset = """
                numChunks = int(ceilf(sequenceLength/float({b_size})));
                blocks = dim3(minibatch, numHeads, numChunks);
"""

    main_block = """
                if (padding_l == {p}) {{
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradOutput.scalar_type(), "dynamicconv_backward", ([&] {{
                        dynamicconv_backward_kernel<{k}, {b_size}, {p}, scalar_t>
                        <<<blocks, {b_size}, 0, stream>>>(
                                    gradOutput.data<scalar_t>(),
                                    input.data<scalar_t>(),
                                    weight.data<scalar_t>(),
                                    minibatch,
                                    sequenceLength,
                                    numFeatures,
                                    numFiltersInBlock,
                                    numHeads,
                                    gradWeight.data<scalar_t>(),
                                    gradInput.data<scalar_t>());
                    }}));
                }} else
"""

    bad_padding = """
                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }
                break;\n
"""

    bad_filter = """
            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping backward pass" << std::endl;
        }
"""

    con_else = """
    } else
"""

    final_else = """
    {
        switch(filterSize) {
"""

    last_return = """
    }
    return {gradInput, gradWeight};
}
"""

    with open("dynamicconv_cuda_backward.cu", "w") as backward:
        backward.write(head)
        for seq in seqs:
            backward.write(sequence_if.format(seq=seq))
            for k, t, m in zip(kernels, thresh, min_block):
                backward.write(case_k.format(k=k))
                if seq <= t:
                    b_size = seq
                else:
                    b_size = m
                    backward.write(chunks_reset.format(b_size=b_size))
                for p in [k // 2, k - 1]:
                    backward.write(main_block.format(k=k, b_size=b_size, p=p))
                backward.write(bad_padding)
            backward.write(bad_filter)
            backward.write(con_else)
        backward.write(final_else)
        for k, m in zip(kernels, min_block):
            backward.write(case_k.format(k=k))
            backward.write(chunks_reset.format(b_size=m))
            for p in [k // 2, k - 1]:
                backward.write(main_block.format(k=k, b_size=m, p=p))
            backward.write(bad_padding)
        backward.write(bad_filter)
        backward.write(last_return)


if __name__ == "__main__":
    gen_forward()
    gen_backward()
