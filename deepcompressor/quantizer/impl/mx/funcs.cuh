//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <stdint.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#define FLOAT32_EXP_BIAS 127
#define FLOAT32_EXP_MAX 255
#define FLOAT32_TRAILING_MBITS 23
#define FLOAT32_IMPLIED1 (1 << FLOAT32_TRAILING_MBITS)
#define FLOAT32_FULL_MBITS (FLOAT32_TRAILING_MBITS + 1)
#define FLOAT32_INF 0x7fe00000
#define FLOAT32_EXP_OFFSET 23
#define FLOAT32_SIGN_OFFSET 31
#define FLOAT32_EXP_MASK 0x7f800000
#define FLOAT32_MANTISSA_MASK 0x007fffff

#define THREADS_PER_BLOCK 32

enum RoundMode {
    ROUND_HALF_TO_EVEN = 8
};

union u_float_int {
    float float_val;
    uint32_t int_val;
};

#ifdef USE_CUDA
__host__ __device__ __forceinline__
#else
inline
#endif
int get_exponent(float f) {
    u_float_int u;
    u.float_val = f;
    u.int_val &= FLOAT32_EXP_MASK;
    return u.int_val >> FLOAT32_TRAILING_MBITS;
}

#ifdef USE_CUDA
__host__ __device__ __forceinline__
#else
inline
#endif
uint32_t get_mantissa(float f) {
    u_float_int u;
    u.float_val = f;
    return u.int_val &= FLOAT32_MANTISSA_MASK;
}

#ifdef USE_CUDA
__host__ __device__ __forceinline__
#else
inline
#endif
uint32_t get_sign(float f) {
    u_float_int u;
    u.float_val = f;
    return u.int_val >> FLOAT32_SIGN_OFFSET;
}

#ifdef USE_CUDA
__host__ __device__ __forceinline__
#else
inline
#endif
uint32_t shift_and_round(uint32_t mantissa, int tail_bits, RoundMode round_mode) {
    if (tail_bits == 0) return mantissa;
    if (tail_bits > 25) return 0;
    uint32_t half = 1 << (tail_bits - 1);
    uint32_t tail = mantissa & ((1 << tail_bits) - 1);
    uint32_t ret = mantissa >> tail_bits;
    if (tail < half) return ret;
    else if (tail > half) return ret + 1;
    else return (ret) % 2 == 1 ? ret + 1 : ret;
}

#ifdef USE_CUDA
__host__ __device__ __forceinline__
#else
inline
#endif
float construct_float(uint32_t sign, uint32_t exponent, uint32_t mantissa) {
    u_float_int u;
    u.int_val = (sign << FLOAT32_SIGN_OFFSET) + (exponent << FLOAT32_EXP_OFFSET) + (mantissa & FLOAT32_MANTISSA_MASK);
    return u.float_val;
}

#ifdef USE_CUDA
__host__ __device__ __forceinline__
#else
inline
#endif
float fake_quantize_element(
    float element,
    float max_norm,
    int ebits,
    int mbits,
    RoundMode round_mode
) {
    int exp = get_exponent(element);
    if (exp == FLOAT32_EXP_MAX) return element;
    int new_bias = (1 << (ebits - 1)) - 1;
    uint32_t mantissa = get_mantissa(element);
    int mantissa_bits = FLOAT32_TRAILING_MBITS;
    if (exp != 0) {
        mantissa = (mantissa | FLOAT32_IMPLIED1);
        mantissa_bits++;
    }

    int new_exp = exp - FLOAT32_EXP_BIAS + new_bias;
    int exp_shift = new_exp > 0 ? 0 : 1 - new_exp;

    int tail_bits = FLOAT32_TRAILING_MBITS - mbits + exp_shift;
    mantissa = shift_and_round(mantissa, tail_bits, round_mode);
    if (mantissa == 0) return 0.0;
    mantissa = mantissa << tail_bits;
    if (mantissa >= (1 << mantissa_bits)) {
        if (exp != 0) mantissa = mantissa >> 1;
        exp++;
    }
    float absolute_ret = construct_float(0, exp, mantissa);
    if (absolute_ret > max_norm) absolute_ret = max_norm;

    u_float_int u;
    u.float_val = absolute_ret;
    u.int_val += (get_sign(element) << FLOAT32_SIGN_OFFSET);
    return u.float_val;
}

void fake_quantize_to_low_precision_fp_cuda(
    float * input,
    float * output,
    uint32_t num_elements,
    int ebits,
    int mbits,
    float max_norm,
    RoundMode round_mode);
