// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureKey.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

class IGenericBatchnormExecutor
{
public:
    virtual ~IGenericBatchnormExecutor() = default;

    // virtual BatchnormSignatureKey signatureKey() const = 0;

    //virtual bool isApplicable(const hipdnn_sdk::data_objects::Node& node) const = 0;

    virtual void batchnormFwdInference(std::any& input,
                                       std::any& scale,
                                       std::any& bias,
                                       std::any& mean,
                                       std::any& variance,
                                       std::any& output,
                                       double epsilon)
        = 0;
};

// template <typename T>
// concept IsBatchnormSignatureRegistryKey = requires {
//     { T::INPUT_DATA_TYPE } -> std::convertible_to<hipdnn_sdk::data_objects::DataType>;
//     { T::SCALE_BIAS_DATA_TYPE } -> std::convertible_to<hipdnn_sdk::data_objects::DataType>;
//     { T::MEAN_VARIANCE_DATA_TYPE } -> std::convertible_to<hipdnn_sdk::data_objects::DataType>;
// };

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

    const hipdnn_sdk::data_objects::DataType INPUT_DATA_TYPE;
    const hipdnn_sdk::data_objects::DataType SCALE_BIAS_DATA_TYPE;
    const hipdnn_sdk::data_objects::DataType MEAN_VARIANCE_DATA_TYPE;

    BatchnormSignatureKey toSignatureKey() const
    {
        return BatchnormSignatureKey{
            INPUT_DATA_TYPE, SCALE_BIAS_DATA_TYPE, MEAN_VARIANCE_DATA_TYPE};
    }
};

template <BatchnormSignatureRegistryKey ThisWontWork>
class BatchnormExecutor : public IGenericBatchnormExecutor
{
public:
    using InputDataType = DataTypeToNative<ThisWontWork.INPUT_DATA_TYPE>;
    using ScaleBiasDataType = DataTypeToNative<ThisWontWork.SCALE_BIAS_DATA_TYPE>;
    using MeanVarianceDataType = DataTypeToNative<ThisWontWork.MEAN_VARIANCE_DATA_TYPE>;

    // BatchnormSignatureKey signatureKey() const
    // {
    //     return {ThisWontWork.INPUT_DATA_TYPE,
    //             ThisWontWork.SCALE_BIAS_DATA_TYPE,
    //             ThisWontWork.MEAN_VARIANCE_DATA_TYPE};
    // }

    // bool isApplicable(const hipdnn_sdk::data_objects::Node& node) const override
    // {
    //     // Implementation to check if this executor can handle the given node
    //     return true; // Placeholder
    // }

    void batchnormFwdInference(std::any& input,
                               std::any& scale,
                               std::any& bias,
                               std::any& mean,
                               std::any& variance,
                               std::any& output,
                               double epsilon) override
    {
        auto& inVar = TensorVariantUtils::unwrapToTensorVariant(input);
        auto& scVar = TensorVariantUtils::unwrapToTensorVariant(scale);
        auto& biVar = TensorVariantUtils::unwrapToTensorVariant(bias);
        auto& meVar = TensorVariantUtils::unwrapToTensorVariant(mean);
        auto& vaVar = TensorVariantUtils::unwrapToTensorVariant(variance);
        auto& outVar = TensorVariantUtils::unwrapToTensorVariant(output);

        auto& inTensor = *std::get<std::unique_ptr<TensorBase<InputDataType>>>(inVar);
        auto& scTensor = *std::get<std::unique_ptr<TensorBase<ScaleBiasDataType>>>(scVar);
        auto& biTensor = *std::get<std::unique_ptr<TensorBase<ScaleBiasDataType>>>(biVar);
        auto& meTensor = *std::get<std::unique_ptr<TensorBase<MeanVarianceDataType>>>(meVar);
        auto& vaTensor = *std::get<std::unique_ptr<TensorBase<MeanVarianceDataType>>>(vaVar);
        auto& outTensor = *std::get<std::unique_ptr<TensorBase<InputDataType>>>(outVar);

        CpuFpReferenceBatchnormImpl<InputDataType, ScaleBiasDataType, MeanVarianceDataType>::
            batchnormFwdInference(
                inTensor, scTensor, biTensor, meTensor, vaTensor, outTensor, epsilon);
    }
};

}
}
