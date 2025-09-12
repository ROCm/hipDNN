// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

template <typename T>
struct BatchnormImpl
{
    static BatchnormImpl getStaticType()
    {
        return T::getStaticType();
    }
};

// 1. Define the signature POD
struct FwdBatchnormSignatureFloat
{
    static constexpr auto INPUT_DATA_TYPE = hipdnn_sdk::data_objects::DataType_FLOAT;
    static constexpr auto NODE_ATTRIBUTES_TYPE
        = hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes;
};
static_assert(BatchnormSignatureDescriptor<FwdBatchnormSignatureFloat>);

struct FwdBatchnormSignatureHalf
{
    static constexpr auto INPUT_DATA_TYPE = hipdnn_sdk::data_objects::DataType_HALF;
    static constexpr auto NODE_ATTRIBUTES_TYPE
        = hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes;
};
static_assert(BatchnormSignatureDescriptor<FwdBatchnormSignatureHalf>);

// 2. Define the key for the registry
struct BatchnormSignatureKey
{
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::NodeAttributes nodeAttributesType;

    bool operator==(const BatchnormSignatureKey& other) const
    {
        return inputDataType == other.inputDataType
               && nodeAttributesType == other.nodeAttributesType;
    }
};

namespace std
{
template <>
struct hash<BatchnormSignatureKey>
{
    std::size_t operator()(const BatchnormSignatureKey& k) const
    {
        return std::hash<int>()(static_cast<int>(k.inputDataType))
               ^ (std::hash<int>()(static_cast<int>(k.nodeAttributesType)) << 1);
    }
};
}

using BatchnormFn
    = std::function<void(std::any&, std::any&, std::any&, std::any&, std::any&, std::any&, double)>;

// Registry keyed by BatchnormSignatureKey
std::unordered_map<BatchnormSignatureKey, BatchnormFn>& batchnormRegistry()
{
    static std::unordered_map<BatchnormSignatureKey, BatchnormFn> _reg;
    return _reg;
}

// Registration helpers for each supported type
struct BatchnormRegistryInitializer
{
    BatchnormRegistryInitializer()
    {
        // FLOAT
        {
            BatchnormSignatureKey key{
                .inputDataType = hipdnn_sdk::data_objects::DataType::DataType_FLOAT,
                .nodeAttributesType
                = hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes,
            };
            batchnormRegistry()[key] = [](std::any& input,
                                          std::any& scale,
                                          std::any& bias,
                                          std::any& mean,
                                          std::any& variance,
                                          std::any& output,
                                          double epsilon) {
                auto& inputT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<float>>>(
                          input)
                          .get();
                auto& scaleT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<float>>>(
                          scale)
                          .get();
                auto& biasT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<float>>>(
                          bias)
                          .get();
                auto& meanT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<float>>>(
                          mean)
                          .get();
                auto& varianceT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<float>>>(
                          variance)
                          .get();
                auto& outputT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<float>>>(
                          output)
                          .get();

                BatchnormBuilder<FwdBatchnormSignatureFloat{}>::Instance::batchnormFwdInference(
                    inputT, scaleT, biasT, meanT, varianceT, outputT, epsilon);
            };
        }
        // HALF
        {
            BatchnormSignatureKey key{
                .inputDataType = hipdnn_sdk::data_objects::DataType::DataType_HALF,
                .nodeAttributesType
                = hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes,
            };
            batchnormRegistry()[key] = [](std::any& input,
                                          std::any& scale,
                                          std::any& bias,
                                          std::any& mean,
                                          std::any& variance,
                                          std::any& output,
                                          double epsilon) {
                auto& inputT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<half>>>(
                          input)
                          .get();
                auto& scaleT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<half>>>(
                          scale)
                          .get();
                auto& biasT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<half>>>(
                          bias)
                          .get();
                auto& meanT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<half>>>(
                          mean)
                          .get();
                auto& varianceT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<half>>>(
                          variance)
                          .get();
                auto& outputT
                    = std::any_cast<std::reference_wrapper<hipdnn_sdk::utilities::Tensor<half>>>(
                          output)
                          .get();

                BatchnormBuilder<FwdBatchnormSignatureHalf{}>::Instance::batchnormFwdInference(
                    inputT, scaleT, biasT, meanT, varianceT, outputT, epsilon);
            };
        }
        // BFLOAT16
        // {
        //     BatchnormSignatureKey key{
        //         .inputDataType = DataType::DataType_BFLOAT16,
        //         .nodeAttributesType = NodeAttributes_BatchnormInferenceAttributes,
        //     };
        //     batchnormRegistry()[key] = [](std::any& input,
        //                                   std::any& scale,
        //                                   std::any& bias,
        //                                   std::any& mean,
        //                                   std::any& variance,
        //                                   std::any& output,
        //                                   double epsilon) {
        //         auto& inputT = std::any_cast<Tensor<bfloat16>&>(input);
        //         auto& scaleT = std::any_cast<Tensor<bfloat16>&>(scale);
        //         auto& biasT = std::any_cast<Tensor<bfloat16>&>(bias);
        //         auto& meanT = std::any_cast<Tensor<bfloat16>&>(mean);
        //         auto& varianceT = std::any_cast<Tensor<bfloat16>&>(variance);
        //         auto& outputT = std::any_cast<Tensor<bfloat16>&>(output);

        //         BatchnormBuilder<FwdBatchnormSignatureBfloat16>::Instance::batchnormFwdInference(
        //             inputT, scaleT, biasT, meanT, varianceT, outputT, epsilon);
        //     };
        // }
    }
};
static BatchnormRegistryInitializer _batchnormRegistryInitializer;
