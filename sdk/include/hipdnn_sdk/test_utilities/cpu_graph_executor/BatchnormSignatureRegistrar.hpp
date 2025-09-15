// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignature.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

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

}
}

//todo, figure out better way to do this. hash cant be inside the hipdnn_sdk namespace
namespace std
{
template <>
struct hash<hipdnn_sdk::test_utilities::BatchnormSignatureKey>
{
    std::size_t operator()(const hipdnn_sdk::test_utilities::BatchnormSignatureKey& k) const
    {
        return std::hash<int>()(static_cast<int>(k.inputDataType))
               ^ (std::hash<int>()(static_cast<int>(k.nodeAttributesType)) << 1);
    }
};
}

namespace hipdnn_sdk
{
namespace test_utilities
{

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
                    = std::any_cast<
                          std::reference_wrapper<hipdnn_sdk::utilities::TensorBase<float>>>(input)
                          .get();
                auto& scaleT
                    = std::any_cast<
                          std::reference_wrapper<hipdnn_sdk::utilities::TensorBase<float>>>(scale)
                          .get();
                auto& biasT
                    = std::any_cast<
                          std::reference_wrapper<hipdnn_sdk::utilities::TensorBase<float>>>(bias)
                          .get();
                auto& meanT
                    = std::any_cast<
                          std::reference_wrapper<hipdnn_sdk::utilities::TensorBase<float>>>(mean)
                          .get();
                auto& varianceT
                    = std::any_cast<
                          std::reference_wrapper<hipdnn_sdk::utilities::TensorBase<float>>>(
                          variance)
                          .get();
                auto& outputT
                    = std::any_cast<
                          std::reference_wrapper<hipdnn_sdk::utilities::TensorBase<float>>>(output)
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
    }
};
static BatchnormRegistryInitializer _batchnormRegistryInitializer;

}
}
