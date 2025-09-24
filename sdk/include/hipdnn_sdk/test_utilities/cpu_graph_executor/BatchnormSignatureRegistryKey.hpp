// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BaseSignatureKey.hpp>

namespace hipdnn_sdk::test_utilities
{

struct BatchnormSignatureRegistryKey
{
    bool operator==(const BatchnormSignatureRegistryKey& other) const
    {
        return inputDataType == other.inputDataType && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType;
    }

    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;
};

constexpr std::array ALL_SUPPORTED_BATCHNORM_SIGNATURES
    = {BatchnormSignatureRegistryKey{.inputDataType = hipdnn_sdk::data_objects::DataType::FLOAT,
                                     .scaleBiasDataType = hipdnn_sdk::data_objects::DataType::FLOAT,
                                     .meanVarianceDataType
                                     = hipdnn_sdk::data_objects::DataType::FLOAT},
       BatchnormSignatureRegistryKey{.inputDataType = hipdnn_sdk::data_objects::DataType::HALF,
                                     .scaleBiasDataType = hipdnn_sdk::data_objects::DataType::HALF,
                                     .meanVarianceDataType
                                     = hipdnn_sdk::data_objects::DataType::HALF}};

}

// // Provide std::hash specialization in std namespace
namespace std
{
template <>
struct hash<hipdnn_sdk::test_utilities::BatchnormSignatureRegistryKey>
{
    std::size_t operator()(const hipdnn_sdk::test_utilities::BatchnormSignatureRegistryKey& k) const
    {
        return std::hash<int>()(static_cast<int>(k.inputDataType)) << 4
               ^ (std::hash<int>()(static_cast<int>(k.scaleBiasDataType)) << 8)
               ^ (std::hash<int>()(static_cast<int>(k.meanVarianceDataType)) << 12);
    }
};
}
