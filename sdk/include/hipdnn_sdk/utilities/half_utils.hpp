// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/amd_detail/amd_hip_fp16.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <string>

inline __HOST_DEVICE__ half operator""_h(long double value)
{
    return {static_cast<float>(value)};
}

namespace std
{
inline __HOST_DEVICE__ half fabs(half num)
{
    return num > 0.0_h ? num : num * -1.0_h;
}

inline __HOST_DEVICE__ half max(half a, half b)
{
    return a > b ? a : b;
}

}

template <>
struct fmt::formatter<half> : fmt::formatter<float>
{
    template <typename FormatContext>
    auto format(half h, FormatContext& ctx) const
    {
        return fmt::formatter<float>::format(static_cast<float>(h), ctx);
    }
};
