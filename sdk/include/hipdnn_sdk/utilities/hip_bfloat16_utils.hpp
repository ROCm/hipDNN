// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <spdlog/fmt/bundled/format.h>


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
struct fmt::formatter<hip_bfloat16> {
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.end();
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    auto format(const hip_bfloat16& h, format_context& ctx) const {
        return format_to(ctx.out(), "{}", static_cast<float>(h));
    }
};
