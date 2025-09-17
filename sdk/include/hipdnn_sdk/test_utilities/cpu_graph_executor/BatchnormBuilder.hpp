// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <functional>

#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
namespace hipdnn_sdk
{
namespace test_utilities
{

template <hipdnn_sdk::data_objects::DataType DT>
struct DataTypeToNative;

template <>
struct DataTypeToNative<hipdnn_sdk::data_objects::DataType::DataType_FLOAT>
{
    using type = float;
};

template <>
struct DataTypeToNative<hipdnn_sdk::data_objects::DataType::DataType_HALF>
{
    using type = half;
};

template <hipdnn_sdk::data_objects::DataType T>
concept BatchnormDataType = (T == hipdnn_sdk::data_objects::DataType::DataType_FLOAT)
                            || (T == hipdnn_sdk::data_objects::DataType::DataType_HALF)
                            || (T == hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16);

template <hipdnn_sdk::data_objects::NodeAttributes T>
concept NodeAttributesType
    = (T == hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes)
      || (T == hipdnn_sdk::data_objects::NodeAttributes_BatchnormBackwardAttributes);

template <auto Sig>
concept ValidBatchnormSignature = requires {
    requires BatchnormDataType<Sig.INPUT_DATA_TYPE>;
    requires BatchnormDataType<Sig.SCALE_BIAS_DATA_TYPE>;
    requires BatchnormDataType<Sig.MEAN_VARIANCE_DATA_TYPE>;
    requires NodeAttributesType<Sig.NODE_ATTRIBUTES_TYPE>;
};

template <typename T>
concept BatchnormSignatureDescriptor = requires(T t) {
    { t.INPUT_DATA_TYPE } -> std::convertible_to<hipdnn_sdk::data_objects::DataType>;
    { t.SCALE_BIAS_DATA_TYPE } -> std::convertible_to<hipdnn_sdk::data_objects::DataType>;
    { t.MEAN_VARIANCE_DATA_TYPE } -> std::convertible_to<hipdnn_sdk::data_objects::DataType>;
    { t.NODE_ATTRIBUTES_TYPE } -> std::convertible_to<hipdnn_sdk::data_objects::NodeAttributes>;
};

template <BatchnormSignatureDescriptor auto SIGNATURE>
    requires ValidBatchnormSignature<SIGNATURE>
struct BatchnormFwdInferenceBuilder
{
    using InputDataType = typename DataTypeToNative<SIGNATURE.INPUT_DATA_TYPE>::type;
    using ScaleBiasDataType = typename DataTypeToNative<SIGNATURE.SCALE_BIAS_DATA_TYPE>::type;
    using MeanVarianceDataType = typename DataTypeToNative<SIGNATURE.MEAN_VARIANCE_DATA_TYPE>::type;

    using Instance = hipdnn_sdk::test_utilities::
        CpuFpReferenceBatchnormImpl<InputDataType, ScaleBiasDataType, MeanVarianceDataType>;
};

}
}
