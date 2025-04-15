// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "data_types_generated.h"
#include "pointwise_attributes_generated.h"
namespace hipdnn_frontend
{

enum class PointwiseMode_t
{
    NOT_SET = 0,
    RELU    = 1,
};

enum class DataType_t
{
    NOT_SET  = 0,
    FLOAT    = 1,
    HALF     = 2,
    BFLOAT16 = 3,
};

static hipdnn::sdk::DataType to_sdk_type(const DataType_t& type)
{
    switch(type)
    {
    case DataType_t::FLOAT:
        return hipdnn::sdk::DataType::DataType_FLOAT;
    case DataType_t::HALF:
        return hipdnn::sdk::DataType::DataType_HALF;
    case DataType_t::BFLOAT16:
        return hipdnn::sdk::DataType::DataType_BFLOAT16;
    default:
        return hipdnn::sdk::DataType::DataType_UNSET;
    }
}

static hipdnn::sdk::PointwiseMode to_sdk_type(const PointwiseMode_t& type)
{
    switch(type)
    {
    case PointwiseMode_t::RELU:
        return hipdnn::sdk::PointwiseMode::PointwiseMode_RELU_FWD;
    default:
        return hipdnn::sdk::PointwiseMode::PointwiseMode_UNSET;
    }
}
}
