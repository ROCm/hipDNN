// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <string>

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

template <>
struct fmt::formatter<hip_bfloat16> : fmt::formatter<float>
{
    template <typename FormatContext>
    auto format(hip_bfloat16 bf, FormatContext& ctx) const
    {
        return fmt::formatter<float>::format(static_cast<float>(bf), ctx);
    }
};
