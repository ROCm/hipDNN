// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/amd_detail/amd_hip_bfloat16.h>

inline __HOST_DEVICE__ hip_bfloat16 operator""_bf(long double value)
{
    return hip_bfloat16(static_cast<float>(value));
}

namespace std
{
inline __HOST_DEVICE__ hip_bfloat16 fabs(hip_bfloat16 num)
{
    return num > 0.0_bf ? num : num * -1.0_bf;
}

inline __HOST_DEVICE__ hip_bfloat16 max(hip_bfloat16 a, hip_bfloat16 b)
{
    return a > b ? a : b;
}
}