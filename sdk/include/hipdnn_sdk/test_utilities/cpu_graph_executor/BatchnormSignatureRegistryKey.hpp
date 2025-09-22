// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_sdk
{
namespace test_utilities
{

struct BatchnormSignatureRegistryKey
{
    constexpr BatchnormSignatureRegistryKey(hipdnn_sdk::data_objects::DataType input,
                                            hipdnn_sdk::data_objects::DataType scaleBias,
                                            hipdnn_sdk::data_objects::DataType meanVariance)
        : INPUT_DATA_TYPE(input)
        , SCALE_BIAS_DATA_TYPE(scaleBias)
        , MEAN_VARIANCE_DATA_TYPE(meanVariance)
    {
    }

    bool operator==(const BatchnormSignatureRegistryKey& other) const
    {
        return INPUT_DATA_TYPE == other.INPUT_DATA_TYPE
               && SCALE_BIAS_DATA_TYPE == other.SCALE_BIAS_DATA_TYPE
               && MEAN_VARIANCE_DATA_TYPE == other.MEAN_VARIANCE_DATA_TYPE;
    }

    const hipdnn_sdk::data_objects::DataType INPUT_DATA_TYPE;
    const hipdnn_sdk::data_objects::DataType SCALE_BIAS_DATA_TYPE;
    const hipdnn_sdk::data_objects::DataType MEAN_VARIANCE_DATA_TYPE;
};

}
}

namespace std
{
template <>
struct hash<hipdnn_sdk::test_utilities::BatchnormSignatureRegistryKey>
{
    std::size_t operator()(const hipdnn_sdk::test_utilities::BatchnormSignatureRegistryKey& k) const
    {
        return std::hash<int>()(static_cast<int>(k.INPUT_DATA_TYPE))
               ^ (std::hash<int>()(static_cast<int>(k.SCALE_BIAS_DATA_TYPE)) << 1)
               ^ (std::hash<int>()(static_cast<int>(k.MEAN_VARIANCE_DATA_TYPE)) << 2);
    }
};
}
