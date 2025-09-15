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
    requires NodeAttributesType<Sig.NODE_ATTRIBUTES_TYPE>;
};

template <typename T>
concept BatchnormSignatureDescriptor = requires(T t) {
    { t.INPUT_DATA_TYPE } -> std::convertible_to<hipdnn_sdk::data_objects::DataType>;
    { t.NODE_ATTRIBUTES_TYPE } -> std::convertible_to<hipdnn_sdk::data_objects::NodeAttributes>;
};

template <BatchnormSignatureDescriptor auto SIGNATURE>
    requires ValidBatchnormSignature<SIGNATURE>
struct BatchnormBuilder
{
    using InputType = typename DataTypeToNative<SIGNATURE.INPUT_DATA_TYPE>::type;

    //todo, add scale biat type
    using Instance = hipdnn_sdk::test_utilities::CpuFpReferenceBatchnormImpl<InputType, InputType>;
};

}
}
