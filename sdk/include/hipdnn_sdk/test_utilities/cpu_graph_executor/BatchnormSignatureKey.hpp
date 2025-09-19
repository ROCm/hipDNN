// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_sdk
{
namespace test_utilities
{

struct BatchnormSignatureKey
{
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;

    bool operator==(const BatchnormSignatureKey& other) const
    {
        return inputDataType == other.inputDataType && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType;
    }
};

}
}

namespace std
{
template <>
struct hash<hipdnn_sdk::test_utilities::BatchnormSignatureKey>
{
    std::size_t operator()(const hipdnn_sdk::test_utilities::BatchnormSignatureKey& k) const
    {
        return std::hash<int>()(static_cast<int>(k.inputDataType))
               ^ (std::hash<int>()(static_cast<int>(k.scaleBiasDataType)) << 1)
               ^ (std::hash<int>()(static_cast<int>(k.meanVarianceDataType)) << 2);
    }
};
}
