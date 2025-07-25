// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_backend_heuristic_type.h>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>

namespace hipdnn_frontend
{

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

}
