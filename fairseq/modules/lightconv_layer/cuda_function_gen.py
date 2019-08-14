# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


def gen_forward():

    kernels = [3, 5, 7, 15, 31, 63, 127, 255]
    seqs = [32 * x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]

    head = """
/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 */

#include "lightconv_cuda.cuh"

std::vector<at::Tensor> lightconv_cuda_forward(at::Tensor input, at::Tensor filters, int padding_l) {

    at::DeviceGuard g(input.device());
    const auto minibatch = input.size(0);
    const auto numFeatures = input.size(1);
    const auto sequenceLength = input.size(2);

    const auto numHeads = filters.size(0);
    const auto filterSize = filters.size(1);

    const auto numFiltersInBlock = numFeatures / numHeads;

    const dim3 blocks(minibatch, numFeatures);

    auto output = at::zeros_like(input);
    auto stream = at::cuda::getCurrentCUDAStream();
"""

    sequence_if = """
    if (sequenceLength <= {seq}) {{
        switch(filterSize) {{
"""

    case_k = """
            case {k}:
"""

    main_block = """
                if (padding_l == {pad}) {{
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {{
                        lightconv_forward_kernel<{k}, {b_size}, {pad}, scalar_t>
                        <<<blocks, {b_size}, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }}));
                }} else
"""

    bad_padding = """
                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;
"""

    bad_filter = """
            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }
"""

    con_else = """
    } else
"""

    final_else = """
    {
        switch(filterSize) {
"""

    final_return = """
    }

    return {output};
}
"""

    with open("lightconv_cuda_forward.cu", 'w') as forward:
        forward.write(head)
        for seq in seqs:
            forward.write(sequence_if.format(seq=seq))
            for k in kernels:
                forward.write(case_k.format(k=k))
                for pad in [k // 2, k - 1]:
                    forward.write(main_block.format(k=k, b_size=seq, pad=pad))
                forward.write(bad_padding)
            forward.write(bad_filter)
            forward.write(con_else)

        forward.write(final_else)
        for k in kernels:
            forward.write(case_k.format(k=k))
            for pad in [k // 2, k - 1]:
                forward.write(main_block.format(k=k, b_size=seq, pad=pad))
            forward.write(bad_padding)
        forward.write(bad_filter)
        forward.write(final_return)


def gen_backward():

    head = """
/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 */

#include "lightconv_cuda.cuh"

std::vector<at::Tensor> lightconv_cuda_backward(
        at::Tensor gradOutput,
        int padding_l,
        at::Tensor input,
        at::Tensor filters) {

    // gradWrtInput
    const int minibatch = input.size(0);
    const int numFeatures = input.size(1);
    const int sequenceLength = input.size(2);

    const int numHeads = filters.size(0);
    const int filterSize = filters.size(1);

    const dim3 gradBlocks(minibatch, numFeatures);
    const dim3 weightGradFirstpassShortBlocks(minibatch, numHeads);
    const dim3 weightGradSecondpassBlocks(numHeads, filterSize);

    const int numFiltersInBlock = numFeatures / numHeads;

    auto gradInput = at::zeros_like(input);
    auto gradFilters = at::zeros_like(filters);

    at::DeviceGuard g(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    switch(filterSize) {
"""

    sequence_if = """
            if (sequenceLength <= {seq}) {{
"""

    case_k = """
        case {k}:
"""

    main_block = """
                if (padding_l == {p}) {{
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {{
                        lightconv_grad_wrt_input_kernel<{k}, {b_size}, {p}, scalar_t>
                        <<<gradBlocks, {b_size}, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());

"""

    weight_grad_short = """
                        at::Tensor tempSumGradFilters = at::zeros({{minibatch, numHeads, filterSize}}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<{k}, {b_size}, {p}, scalar_t>
                        <<<weightGradFirstpassShortBlocks, {b_size}, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<{k}, {b_size}, scalar_t>
                        <<<weightGradSecondpassBlocks, {b_size}, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }}));
                }} else
"""

    weight_grad = """
                        at::Tensor tempSumGradFilters = at::zeros({{minibatch, numFeatures, filterSize}}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<{k}, {b_size}, {p}, scalar_t>
                        <<<gradBlocks, {b_size}, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<{k}, {b_size}, scalar_t>
                        <<<weightGradSecondpassBlocks, {b_size}, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }}));
                }} else
"""

    bad_padding = """
                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }
"""

    breakout = """
                break;
"""

    bad_filter = """
        default:
            std::cout << "WARNING: Unsupported filter length passed - skipping backward pass" << std::endl;
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
    return {gradInput, gradFilters};
}
"""

    kernels = [3, 5, 7, 15, 31, 63, 127, 255]
    seqs = [32 * x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
    thresh = [32, 32, 64, 128, 256, -1, -1, -1]
    max_mem = [-1, -1, -1, -1, -1, 192, 96, 64]

    with open("lightconv_cuda_backward.cu", 'w') as backward:
        backward.write(head)
        for (k, t, mem) in zip(kernels, thresh, max_mem):
            backward.write(case_k.format(k=k))
            for seq in seqs:
                if (t == -1 or seq <= t) and (mem == -1 or seq < mem):
                    backward.write(sequence_if.format(seq=seq))
                    for p in [k // 2, k - 1]:
                        backward.write(main_block.format(k=k, b_size=seq, p=p))
                        backward.write(weight_grad_short.format(k=k, b_size=seq, p=p))
                    backward.write(bad_padding)
                else:
                    for p in [k // 2, k - 1]:
                        backward.write(main_block.format(k=k, b_size=32, p=p))
                        backward.write(weight_grad.format(k=k, b_size=32, p=p))
                    backward.write(bad_padding)
                    backward.write(breakout)
                    break
                backward.write(con_else)
        backward.write(bad_filter)
        backward.write(last_return)


if __name__ == "__main__":
    gen_forward()
    gen_backward()
