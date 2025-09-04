// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_bfloat16.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <string>

#define HIPRT_NAN_BF16 __ushort_as_bfloat16(static_cast<unsigned short>(0x7FFFU))

inline __HOST_DEVICE__ hip_bfloat16 __ushort_as_bfloat16(const unsigned short int a)
{
    hip_bfloat16 val;
    val.data = a;
    return val;
}

inline __HOST_DEVICE__ hip_bfloat16 operator""_bf(long double value)
{
    return hip_bfloat16(static_cast<float>(value));
}

inline __HOST_DEVICE__ hip_bfloat16 __habs(const hip_bfloat16 a)
{
    hip_bfloat16 abs = a;
    abs.data &= 0x7FFF;
    return abs;
}

inline __HOST_DEVICE__ bool __hisnan(const hip_bfloat16 a)
{
    hip_bfloat16 hr = a;
    return !(~hr.data & 0x7f80) && +(hr.data & 0x7f);
}

//#define HIPRT_NAN_BF16 __ushort_as_bfloat16((unsigned short)0x7FFFU)

inline __HOST_DEVICE__ hip_bfloat16 __hmax(const hip_bfloat16 a, const hip_bfloat16 b)
{
    auto a_nan = __hisnan(a), b_nan = __hisnan(b);
    if(a_nan || b_nan)
    {
        if(a_nan && b_nan)
            return HIPRT_NAN_BF16; // return canonical NaN
        return a_nan ? b : a;
    }
    return a.data > b.data ? a : b;
}

namespace std
{
inline __HOST_DEVICE__ hip_bfloat16 fabs(hip_bfloat16 num)
{
    return __habs(num);
}

inline __HOST_DEVICE__ hip_bfloat16 max(hip_bfloat16 a, hip_bfloat16 b)
{
    return __hmax(a, b);
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
