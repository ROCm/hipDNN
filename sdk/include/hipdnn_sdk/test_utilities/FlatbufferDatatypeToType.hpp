// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

namespace
{
template <typename T>
struct TypeWrapper
{
    using type = T;
};

// Mapping function
template <hipdnn_sdk::data_objects::DataType DT>
constexpr auto getTypeWrapper()
{
    using DataType = hipdnn_sdk::data_objects::DataType;

    if constexpr(DT == DataType::FLOAT)
    {
        return TypeWrapper<float>{};
    }
    else if constexpr(DT == DataType::HALF)
    {
        return TypeWrapper<half>{};
    }
    else if constexpr(DT == DataType::DOUBLE)
    {
        return TypeWrapper<double>{};
    }
    else if constexpr(DT == DataType::INT32)
    {
        return TypeWrapper<int32_t>{};
    }
    else if constexpr(DT == DataType::BFLOAT16)
    {
        return TypeWrapper<hip_bfloat16>{};
    }
    else
    {
        static_assert(DT != DT, "Unsupported DataType");
    }
}
}

namespace hipdnn_sdk
{
namespace test_utilities
{

template <hipdnn_sdk::data_objects::DataType DT>
struct DataTypeToNative
{
    using type = typename decltype(getTypeWrapper<DT>())::type;
};

template <hipdnn_sdk::data_objects::DataType DT>
using DataTypeToNative_t = typename DataTypeToNative<DT>::type;

}
}
