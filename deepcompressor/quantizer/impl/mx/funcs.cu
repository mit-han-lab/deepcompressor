//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <c10/cuda/CUDAGuard.h>
#include "funcs.cuh"

__global__ void fake_quantize_kernel(
    float * input,
    float * output,
    uint32_t num_elements,
    int ebits,
    int mbits,
    float max_norm,
    RoundMode round_mode
) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_elements) {
        return;
    }
    output[index] = fake_quantize_element(input[index], max_norm, ebits, mbits, round_mode);
}

void fake_quantize_to_low_precision_fp_cuda(
    float * input,
    float * output,
    uint32_t num_elements,
    int ebits,
    int mbits,
    float max_norm,
    RoundMode round_mode
) {
    int blocks = num_elements % THREADS_PER_BLOCK == 0 ? num_elements / THREADS_PER_BLOCK : num_elements / THREADS_PER_BLOCK + 1;
    fake_quantize_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input, output, num_elements, ebits, mbits, max_norm, round_mode
    );
}
