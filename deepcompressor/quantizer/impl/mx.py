import struct
import torch

# Constants based on the provided macros
FLOAT32_EXP_BIAS = 127
FLOAT32_EXP_MAX = 255
FLOAT32_TRAILING_MBITS = 23
FLOAT32_IMPLIED1 = (1 << FLOAT32_TRAILING_MBITS)
FLOAT32_FULL_MBITS = (FLOAT32_TRAILING_MBITS + 1)
FLOAT32_INF = 0x7fe00000
FLOAT32_EXP_OFFSET = 23
FLOAT32_SIGN_OFFSET = 31
FLOAT32_EXP_MASK = 0x7f800000
FLOAT32_MANTISSA_MASK = 0x007fffff

# RoundMode is assumed to be an integer, you can define it based on your specific use case.
RoundMode = int  # This can be further defined if you have specific enum values for rounding modes.

from torch.utils.cpp_extension import load
import torch

import os
os.environ['PATH'] = '/usr/lib/cuda/bin:' + os.environ['PATH']

extra_cflags = ["-DUSE_CUDA"]
extra_cuda_cflags = ["-DUSE_CUDA"]

mx = load(name="mx",
               sources=[
                        "deepcompressor/quantizer/impl/mx/funcs.cpp",
                        "deepcompressor/quantizer/impl/mx/funcs.cu"
                        ],
               extra_cuda_cflags=extra_cuda_cflags,
               extra_cflags=extra_cflags,
               extra_include_paths=[
                   "/group/amdneuralopt/zhaofeng/tools/miniconda3/envs/deepcompressor/lib/python3.12/site-packages/triton/backends/nvidia/include/",
                   ],
               verbose=True)


def get_dtype_params(dtype: str) -> tuple[int, int, int]:
    if dtype == "fp6_e3m2_all":
        ebits, mbits = 3, 2
        emax = 2**(ebits - 1)
    elif dtype == "sfp6_e2m3_all":
        ebits, mbits = 2, 3
        emax = 2**(ebits - 1)
    elif dtype == "sfp4_e2m1_all":
        ebits, mbits = 2, 1
        emax = 2**(ebits - 1)
    else:
        raise Exception("Unknown element format %s" % dtype)

    return ebits, mbits, emax



def fake_quantize_mx(input_tensor, scale, element_dtype):
    ebits, mbits, _ = get_dtype_params(element_dtype)
    max_exp = pow(2.0, ebits) - 1
    offset_exp = pow(2.0, ebits - 1) - 1
    quant_max = pow(2.0, max_exp - offset_exp) * (1 + (pow(2.0, mbits) - 1) / (pow(2.0, mbits)))

    input_tensor = input_tensor / scale

    output_tensor = mx.fake_quantize_to_low_precision_fp(input_tensor.contiguous(), ebits, mbits, quant_max, 0)

    return output_tensor
