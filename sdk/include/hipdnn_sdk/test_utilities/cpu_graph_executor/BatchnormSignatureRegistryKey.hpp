// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BaseSignatureKey.hpp>

namespace hipdnn_sdk::test_utilities
{

struct BatchnormSignatureRegistryKey : public BaseSigKey
{
    constexpr BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType input,
                                            hipdnn_sdk::data_objects::DataType scaleBias,
                                            hipdnn_sdk::data_objects::DataType meanVariance)
        : inputDataType(input)
        , scaleBiasDataType(scaleBias)
        , meanVarianceDataType(meanVariance)
    {
        nodeType = hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes;
    }

    bool operator==(const BatchnormSignatureRegistryKey& other) const
    {
        return nodeType == other.nodeType && inputDataType == other.inputDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType;
    }

    std::size_t operator()(const BatchnormSignatureRegistryKey& k) const
    {
        return std::hash<int>()(static_cast<int>(k.nodeType))
               ^ std::hash<int>()(static_cast<int>(k.inputDataType)) << 4
               ^ (std::hash<int>()(static_cast<int>(k.scaleBiasDataType)) << 8)
               ^ (std::hash<int>()(static_cast<int>(k.meanVarianceDataType)) << 12);
    }

    size_t hash_self() const override
    {
        return (*this)(*this);
    }

    bool equal(const BaseSigKey& key) const override
    {
        return *this == dynamic_cast<const BatchnormSignatureRegistryKey&>(key);
    }

    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;
};

constexpr std::array ALL_SUPPORTED_BATCHNORM_SIGNATURES
    = {BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType::FLOAT,
                                     hipdnn_sdk::data_objects::DataType::FLOAT,
                                     hipdnn_sdk::data_objects::DataType::FLOAT),
       BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType::HALF,
                                     hipdnn_sdk::data_objects::DataType::HALF,
                                     hipdnn_sdk::data_objects::DataType::HALF)};

}