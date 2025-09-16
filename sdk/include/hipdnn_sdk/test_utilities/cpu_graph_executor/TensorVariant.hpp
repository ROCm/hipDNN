// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

#pragma once

namespace hipdnn_sdk
{
namespace test_utilities
{

using TensorVariant
    = std::variant<std::unique_ptr<TensorBase<float>>, std::unique_ptr<TensorBase<half>>>;

template <typename T>
static std::unique_ptr<TensorBase<T>> createHostOnlyShallowTensor(
    void* ptr, const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    return std::make_unique<ShallowTensor<T>>(ptr, dims, strides);
}

static TensorVariant
    createHostOnlyShallowTensorVariantInternal(hipdnn_sdk::data_objects::DataType dataType,
                                               void* ptr,
                                               const std::vector<int64_t>& dims,
                                               const std::vector<int64_t>& strides)
{
    switch(dataType)
    {
    case hipdnn_sdk::data_objects::DataType::DataType_FLOAT:
        return createHostOnlyShallowTensor<float>(ptr, dims, strides);
    case hipdnn_sdk::data_objects::DataType::DataType_HALF:
        return createHostOnlyShallowTensor<half>(ptr, dims, strides);
    case hipdnn_sdk::data_objects::DataType::DataType_UNSET:
    case hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16:
    case hipdnn_sdk::data_objects::DataType::DataType_DOUBLE:
    case hipdnn_sdk::data_objects::DataType::DataType_UINT8:
    case hipdnn_sdk::data_objects::DataType::DataType_INT32:
    default:
        break;
    }

    throw std::runtime_error("Unsupported data type for shallow tensor creation");
}

static std::vector<int64_t> flatbufferVectorToStd(const ::flatbuffers::Vector<int64_t>* fbVec)
{
    std::vector<int64_t> result;
    if(fbVec == nullptr)
    {
        return result;
    }
    result.reserve(fbVec->size());
    for(auto v : *fbVec)
    {
        result.push_back(v);
    }
    return result;
}

static TensorVariant createHostOnlyShallowTensorVariant(
    const hipdnn_sdk::data_objects::TensorAttributes& tensorAttributes, void* ptr)
{
    return createHostOnlyShallowTensorVariantInternal(
        tensorAttributes.data_type(),
        ptr,
        flatbufferVectorToStd(tensorAttributes.dims()),
        flatbufferVectorToStd(tensorAttributes.strides()));
}

}
}
