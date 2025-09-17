// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBuilder.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

struct BatchnormSignatureKey
{
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_sdk::data_objects::DataType meanVarianceDataType;

    hipdnn_sdk::data_objects::NodeAttributes nodeAttributesType;

    bool operator==(const BatchnormSignatureKey& other) const
    {
        return inputDataType == other.inputDataType && scaleBiasDataType == other.scaleBiasDataType
               && meanVarianceDataType == other.meanVarianceDataType
               && nodeAttributesType == other.nodeAttributesType;
    }
};

/**
 * How to add a new batchnorm signature
 * 
 * To add a new batchnorm signature:
 * 1. Use the DEFINE_BATCHNORM_SIGNATURE macro with your custom name and data types:
 *    DEFINE_BATCHNORM_SIGNATURE(YourSignatureName,
 *                               InputDataType,
 *                               ScaleBiasDataType, 
 *                               MeanVarianceDataType,
 *                               NodeAttributesType);
 * 
 * 2. Update the BatchnormSignatureVariants variant type to include your new signature:
 *    using BatchnormSignatureVariants = std::variant<
 *        FwdBatchnormSignatureFloat,
 *        FwdBatchnormSignatureHalf,
 *        FwdBatchnormSignatureTest,
 *        YourSignatureName>;  // Add your new type here
 * 
 * If you forget to update the BatchnormSignatureVariants, the graph will not be able to find
 * your new signature. 
 */
#define DEFINE_BATCHNORM_SIGNATURE(Name, InputType, ScaleBiasType, MeanVarianceType, NodeAttrType) \
    struct Name                                                                                    \
    {                                                                                              \
        static constexpr auto INPUT_DATA_TYPE = hipdnn_sdk::data_objects::InputType;               \
        static constexpr auto SCALE_BIAS_DATA_TYPE = hipdnn_sdk::data_objects::ScaleBiasType;      \
        static constexpr auto MEAN_VARIANCE_DATA_TYPE                                              \
            = hipdnn_sdk::data_objects::MeanVarianceType;                                          \
        static constexpr auto NODE_ATTRIBUTES_TYPE = hipdnn_sdk::data_objects::NodeAttrType;       \
    };                                                                                             \
    static_assert(BatchnormSignatureDescriptor<Name>)

DEFINE_BATCHNORM_SIGNATURE(FwdBatchnormSignatureFloat,
                           DataType::FLOAT,
                           DataType::FLOAT,
                           DataType::FLOAT,
                           NodeAttributes::BatchnormInferenceAttributes);

DEFINE_BATCHNORM_SIGNATURE(FwdBatchnormSignatureHalf,
                           DataType::HALF,
                           DataType::HALF,
                           DataType::HALF,
                           NodeAttributes::BatchnormInferenceAttributes);

using BatchnormSignatureVariants
    = std::variant<FwdBatchnormSignatureFloat, FwdBatchnormSignatureHalf>;

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
               ^ (std::hash<int>()(static_cast<int>(k.meanVarianceDataType)) << 2)
               ^ (std::hash<int>()(static_cast<int>(k.nodeAttributesType)) << 3);
    }
};
}
