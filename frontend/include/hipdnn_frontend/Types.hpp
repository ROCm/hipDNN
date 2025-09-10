// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <HipdnnBackendHeuristicType.h>
#include <hipdnn_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_frontend
{

enum class ConvolutionMode
{
    NOT_SET = 0,
    CROSS_CORRELATION = 1,
    CONVOLUTION = 2
};
typedef ConvolutionMode ConvolutionMode_t; // NOLINT(readability-identifier-naming)

enum class PointwiseMode
{
    NOT_SET = 0,
    RELU_FWD = 1,
};
typedef PointwiseMode PointwiseMode_t; // NOLINT(readability-identifier-naming)

enum class DataType
{
    NOT_SET = 0,
    FLOAT = 1,
    HALF = 2,
    BFLOAT16 = 3,
    DOUBLE = 4,
    UINT8 = 5,
    INT32 = 6,
};
typedef DataType DataType_t; // NOLINT(readability-identifier-naming)

enum class HeuristicMode
{
    FALLBACK,
};
typedef HeuristicMode HeurMode_t; // NOLINT(readability-identifier-naming)

template <typename T>
DataType getDataTypeEnumFromType()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return DataType::HALF;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return DataType::BFLOAT16;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return DataType::DOUBLE;
    }
    else if constexpr(std::is_same_v<T, uint8_t>)
    {
        return DataType::UINT8;
    }
    else if constexpr(std::is_same_v<T, int32_t>)
    {
        return DataType::INT32;
    }
    else
    {
        return DataType::NOT_SET;
    }
}

inline hipdnn_sdk::data_objects::ConvMode toSdkType(const ConvolutionMode& type)
{
    switch(type)
    {
    case ConvolutionMode::CROSS_CORRELATION:
        return hipdnn_sdk::data_objects::ConvMode::ConvMode_CROSS_CORRELATION;
    case ConvolutionMode::CONVOLUTION:
        return hipdnn_sdk::data_objects::ConvMode::ConvMode_CONVOLUTION;
    default:
        return hipdnn_sdk::data_objects::ConvMode::ConvMode_UNSET;
    }
}

inline hipdnn_sdk::data_objects::DataType toSdkType(const DataType& type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return hipdnn_sdk::data_objects::DataType::DataType_FLOAT;
    case DataType::HALF:
        return hipdnn_sdk::data_objects::DataType::DataType_HALF;
    case DataType::BFLOAT16:
        return hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16;
    case DataType::DOUBLE:
        return hipdnn_sdk::data_objects::DataType::DataType_DOUBLE;
    case DataType::UINT8:
        return hipdnn_sdk::data_objects::DataType::DataType_UINT8;
    case DataType::INT32:
        return hipdnn_sdk::data_objects::DataType::DataType_INT32;
    default:
        return hipdnn_sdk::data_objects::DataType::DataType_UNSET;
    }
}

inline hipdnn_sdk::data_objects::PointwiseMode toSdkType(const PointwiseMode& type)
{
    switch(type)
    {
    case PointwiseMode::RELU_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_RELU_FWD;
    default:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_UNSET;
    }
}

inline hipdnnBackendHeurMode_t toBackendType(const HeuristicMode& type)
{
    switch(type)
    {
    case HeuristicMode::FALLBACK:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    default:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    }
}

// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const DataType& type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return "fp32";
    case DataType::HALF:
        return "fp16";
    case DataType::BFLOAT16:
        return "bf16";
    case DataType::DOUBLE:
        return "fp64";
    case DataType::UINT8:
        return "uint8";
    case DataType::INT32:
        return "int32";
    default:
        return "unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, const DataType& type)
{
    return os << to_string(type);
}

}
