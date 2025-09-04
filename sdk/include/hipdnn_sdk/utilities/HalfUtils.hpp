// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_fp16.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <string>

inline __HOST_DEVICE__ half __ushort_as_half(unsigned short x)
{
    __half_raw r;
    r.x = x;
    return r;
}

inline __HOST_DEVICE__ half operator""_h(long double value)
{
    return {static_cast<float>(value)};
}

inline __HOST_DEVICE__ half __habs(__half num)
{
    auto raw = static_cast<__half_raw>(num);
    raw.x &= 0x7FFFu;
    return raw;
}

inline __HOST_DEVICE__ bool __hisnan(__half x)
{
    __half_raw hr = x;
    return (hr.x & 0x7FFFU) > 0x7C00u;
}

inline __HOST_DEVICE__ __half __hmax(const __half a, const __half b)
{
    if(__hisnan(a) && !__hisnan(b))
        return b;
    if(!__hisnan(a) && __hisnan(b))
        return a;
    if(__hisnan(a) && __hisnan(b))
        return HIPRT_NAN_FP16;
    if(static_cast<__half_raw>(a).x > static_cast<__half_raw>(b).x)
        return __half_raw{static_cast<__half_raw>(a).x};
    return __half_raw{static_cast<__half_raw>(b).x};
}

namespace std
{
inline __HOST_DEVICE__ half fabs(half num)
{
    return __habs(num);
}

inline __HOST_DEVICE__ half max(half a, half b)
{
    return __hmax(a, b);
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
