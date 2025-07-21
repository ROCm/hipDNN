// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/amd_detail/amd_hip_fp16.h>

__HOST_DEVICE__ half operator"" _h(long double value)
{
    return half(static_cast<float>(value));
}

namespace std
{
    __HOST_DEVICE__ half fabs(half num)
    {
        return num > 0.0_h ? num : num * -1.0_h;
    }

    __HOST_DEVICE__ half max(half a, half b)
    {
        return a > b ? a : b;
    }

}

