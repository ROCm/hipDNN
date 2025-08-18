// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_backend_heuristic_type.h>
#include <hipdnn_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/utilities/half_utils.hpp>
#include <hipdnn_sdk/utilities/hip_bfloat16_utils.hpp>

namespace hipdnn_frontend
{

enum class ConvolutionMode_t
{
    NOT_SET = 0,
    CROSS_CORRELATION = 1,
    CONVOLUTION = 2
};

enum class PointwiseMode_t
{
    NOT_SET = 0,
    RELU_FWD = 1,
};

enum class DataType_t
{
    NOT_SET = 0,
    FLOAT = 1,
    HALF = 2,
    BFLOAT16 = 3,
    DOUBLE = 4,
    UINT8 = 5,
    INT32 = 6,
};

enum class HeurMode_t
{
    FALLBACK,
};

template <typename T>
DataType_t get_data_type_enum_from_type()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return DataType_t::FLOAT;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return DataType_t::HALF;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return DataType_t::BFLOAT16;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return DataType_t::DOUBLE;
    }
    else if constexpr(std::is_same_v<T, uint8_t>)
    {
        return DataType_t::UINT8;
    }
    else if constexpr(std::is_same_v<T, int32_t>)
    {
        return DataType_t::INT32;
    }
    else
    {
        return DataType_t::NOT_SET;
    }
}

[[maybe_unused]] static hipdnn_sdk::data_objects::ConvMode
    to_sdk_type(const ConvolutionMode_t& type)
{
    switch(type)
    {
    case ConvolutionMode_t::CROSS_CORRELATION:
        return hipdnn_sdk::data_objects::ConvMode::ConvMode_CROSS_CORRELATION;
    case ConvolutionMode_t::CONVOLUTION:
        return hipdnn_sdk::data_objects::ConvMode::ConvMode_CONVOLUTION;
    default:
        return hipdnn_sdk::data_objects::ConvMode::ConvMode_UNSET;
    }
}

[[maybe_unused]] static hipdnn_sdk::data_objects::DataType to_sdk_type(const DataType_t& type)
{
    switch(type)
    {
    case DataType_t::FLOAT:
        return hipdnn_sdk::data_objects::DataType::DataType_FLOAT;
    case DataType_t::HALF:
        return hipdnn_sdk::data_objects::DataType::DataType_HALF;
    case DataType_t::BFLOAT16:
        return hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16;
    case DataType_t::DOUBLE:
        return hipdnn_sdk::data_objects::DataType::DataType_DOUBLE;
    case DataType_t::UINT8:
        return hipdnn_sdk::data_objects::DataType::DataType_UINT8;
    case DataType_t::INT32:
        return hipdnn_sdk::data_objects::DataType::DataType_INT32;
    default:
        return hipdnn_sdk::data_objects::DataType::DataType_UNSET;
    }
}

[[maybe_unused]] static hipdnn_sdk::data_objects::PointwiseMode
    to_sdk_type(const PointwiseMode_t& type)
{
    switch(type)
    {
    case PointwiseMode_t::RELU_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_RELU_FWD;
    default:
        return hipdnn_sdk::data_objects::PointwiseMode::PointwiseMode_UNSET;
    }
}

[[maybe_unused]] static hipdnnBackendHeurMode_t to_backend_type(const HeurMode_t& type)
{
    switch(type)
    {
    case HeurMode_t::FALLBACK:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    default:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    }
}

[[maybe_unused]] static const char* to_string(const DataType_t& type)
{
    switch(type)
    {
    case DataType_t::FLOAT:
        return "fp32";
    case DataType_t::HALF:
        return "fp16";
    case DataType_t::BFLOAT16:
        return "bf16";
    case DataType_t::DOUBLE:
        return "fp64";
    case DataType_t::UINT8:
        return "uint8";
    case DataType_t::INT32:
        return "int32";
    default:
        return "unknown";
    }
}

[[maybe_unused]] static std::ostream& operator<<(std::ostream& os, const DataType_t& type)
{
    return os << to_string(type);
}

}
