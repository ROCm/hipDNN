// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/amd_detail/amd_hip_fp16.h>
#include <spdlog/fmt/bundled/format.h>

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
struct fmt::formatter<half> {
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.end();
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    auto format(const half& h, format_context& ctx) const {
        return format_to(ctx.out(), "{}", static_cast<float>(h));
    }
};
