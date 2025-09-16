// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBuilder.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureRegistrar.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

class TestCpuReferenceGraphExecutor
{
private:
public:
    static flatbuffers::FlatBufferBuilder createValidBatchnormGraph(
        std::vector<int64_t> strides = {1, 3, 224, 224},
        std::vector<int64_t> dims = {1, 3, 224, 224},
        bool hasOptionalAttributes = true,
        hipdnn_sdk::data_objects::DataType inputDataType = DataType_FLOAT,
        hipdnn_sdk::data_objects::DataType scaleBiasDataType = DataType_FLOAT,
        hipdnn_sdk::data_objects::DataType meanVarianceDataType = DataType_FLOAT)
    {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
            tensorAttributes;

        std::vector<int64_t> derivedStrides = {1, strides[1], 1, 1};
        std::vector<int64_t> derivedDims = {1, dims[1], 1, 1};

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 1, "x", inputDataType, &strides, &dims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 2, "y", inputDataType, &strides, &dims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 3, "scale", scaleBiasDataType, &derivedStrides, &derivedDims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 4, "bias", scaleBiasDataType, &derivedStrides, &derivedDims));

        if(hasOptionalAttributes)
        {
            tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 5, "est_mean", meanVarianceDataType, &derivedStrides, &derivedDims));

            tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 6, "est_variance", meanVarianceDataType, &derivedStrides, &derivedDims));
        }

        auto bnormAttributes = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(
            builder,
            1, // x uid
            hasOptionalAttributes ? flatbuffers::Optional<int64_t>(5)
                                  : flatbuffers::nullopt, // mean uid
            hasOptionalAttributes ? flatbuffers::Optional<int64_t>(6)
                                  : flatbuffers::nullopt, // inv_variance uid
            3, // scale uid
            4, // bias uid
            2 // y uid
        );

        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
        auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            "batchnorm",
            hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes,
            bnormAttributes.Union());
        nodes.push_back(node);

        auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                       "test",
                                                                       DataType_FLOAT,
                                                                       DataType_HALF,
                                                                       DataType_BFLOAT16,
                                                                       &tensorAttributes,
                                                                       &nodes);
        builder.Finish(graphOffset);
        return builder;
    }
};

template <typename T>
hipdnnPluginDeviceBuffer_t
    generateRandomHostBuffer(TensorBase<T>& tensor, int uid, T min, T max, unsigned int seed = 0)
{
    tensor.fillWithRandomValues(min, max, seed);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().hostData();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t generateEmptyHostBuffer(TensorBase<T>& tensor, int uid)
{
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().hostData();
    return buffer;
}

TEST(TestCpuReferenceGraphExecutor, CanExecuteAGraphWithAFwdBatchnormNode)
{
    unsigned int seed = std::random_device{}();

    std::vector<int64_t> dims = {1, 3, 14, 14};

    std::vector<int64_t> derivedDims = {1, dims[1]};

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

    using InputType = float;
    using ScaleBiasType = float;
    using MeanVarianceType = float;
    TensorLayout layout = TensorLayout::NCHW;

    PinnedTensor<InputType> xTensor(dims, layout);
    deviceBuffers.push_back(generateRandomHostBuffer(
        xTensor, 1, static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed));

    PinnedTensor<InputType> yTensor(dims, layout);
    deviceBuffers.push_back(generateEmptyHostBuffer(yTensor, 2));

    PinnedTensor<ScaleBiasType> scaleTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(
        scaleTensor, 3, static_cast<ScaleBiasType>(0.0f), static_cast<ScaleBiasType>(1.0f), seed));

    PinnedTensor<ScaleBiasType> biasTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(
        biasTensor, 4, static_cast<ScaleBiasType>(0.0f), static_cast<ScaleBiasType>(1.0f), seed));

    PinnedTensor<MeanVarianceType> meanTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(meanTensor,
                                                     5,
                                                     static_cast<MeanVarianceType>(0.0f),
                                                     static_cast<MeanVarianceType>(1.0f),
                                                     seed));

    PinnedTensor<MeanVarianceType> varianceTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(varianceTensor,
                                                     6,
                                                     static_cast<MeanVarianceType>(0.1f),
                                                     static_cast<MeanVarianceType>(1.0f),
                                                     seed));

    auto batchnormBuilder = TestCpuReferenceGraphExecutor::createValidBatchnormGraph(
        xTensor.strides(),
        xTensor.dims(),
        true,
        hipdnn_sdk::data_objects::DataType_FLOAT,
        hipdnn_sdk::data_objects::DataType_FLOAT,
        hipdnn_sdk::data_objects::DataType_FLOAT);

    auto batchnormGraph = batchnormBuilder.GetBufferPointer();

    std::unordered_map<int64_t, void*> variantPack;
    for(const auto& deviceBuffer : deviceBuffers)
    {
        variantPack[deviceBuffer.uid] = deviceBuffer.ptr;
    }

    hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor::execute(
        batchnormGraph, batchnormBuilder.GetSize(), variantPack);
}

TEST(TestCpuReferenceGraphExecutor, CanExecuteAGraphWithAFwdBatchnormNodeHalfsies)
{
    unsigned int seed = std::random_device{}();

    std::vector<int64_t> dims = {1, 3, 14, 14};

    std::vector<int64_t> derivedDims = {1, dims[1]};

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

    using InputType = half;
    using ScaleBiasType = half;
    using MeanVarianceType = half;
    TensorLayout layout = TensorLayout::NCHW;

    PinnedTensor<InputType> xTensor(dims, layout);
    deviceBuffers.push_back(generateRandomHostBuffer(
        xTensor, 1, static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed));

    PinnedTensor<InputType> yTensor(dims, layout);
    deviceBuffers.push_back(generateEmptyHostBuffer(yTensor, 2));

    PinnedTensor<ScaleBiasType> scaleTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(
        scaleTensor, 3, static_cast<ScaleBiasType>(0.0f), static_cast<ScaleBiasType>(1.0f), seed));

    PinnedTensor<ScaleBiasType> biasTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(
        biasTensor, 4, static_cast<ScaleBiasType>(0.0f), static_cast<ScaleBiasType>(1.0f), seed));

    PinnedTensor<MeanVarianceType> meanTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(meanTensor,
                                                     5,
                                                     static_cast<MeanVarianceType>(0.0f),
                                                     static_cast<MeanVarianceType>(1.0f),
                                                     seed));

    PinnedTensor<MeanVarianceType> varianceTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(varianceTensor,
                                                     6,
                                                     static_cast<MeanVarianceType>(0.1f),
                                                     static_cast<MeanVarianceType>(1.0f),
                                                     seed));

    auto batchnormBuilder = TestCpuReferenceGraphExecutor::createValidBatchnormGraph(
        xTensor.strides(),
        xTensor.dims(),
        true,
        hipdnn_sdk::data_objects::DataType_HALF,
        hipdnn_sdk::data_objects::DataType_HALF,
        hipdnn_sdk::data_objects::DataType_HALF);

    auto batchnormGraph = batchnormBuilder.GetBufferPointer();

    std::unordered_map<int64_t, void*> variantPack;
    for(const auto& deviceBuffer : deviceBuffers)
    {
        variantPack[deviceBuffer.uid] = deviceBuffer.ptr;
    }

    hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor::execute(
        batchnormGraph, batchnormBuilder.GetSize(), variantPack);
}

TEST(TestCpuReferenceGraphExecutor, CanExecuteAGraphWithAFwdBatchnormNodeTypesDontExist)
{
    unsigned int seed = std::random_device{}();

    std::vector<int64_t> dims = {1, 3, 14, 14};

    std::vector<int64_t> derivedDims = {1, dims[1]};

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

    using InputType = float;
    using ScaleBiasType = half;
    using MeanVarianceType = half;
    TensorLayout layout = TensorLayout::NCHW;

    PinnedTensor<InputType> xTensor(dims, layout);
    deviceBuffers.push_back(generateRandomHostBuffer(
        xTensor, 1, static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed));

    PinnedTensor<InputType> yTensor(dims, layout);
    deviceBuffers.push_back(generateEmptyHostBuffer(yTensor, 2));

    PinnedTensor<ScaleBiasType> scaleTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(
        scaleTensor, 3, static_cast<ScaleBiasType>(0.0f), static_cast<ScaleBiasType>(1.0f), seed));

    PinnedTensor<ScaleBiasType> biasTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(
        biasTensor, 4, static_cast<ScaleBiasType>(0.0f), static_cast<ScaleBiasType>(1.0f), seed));

    PinnedTensor<MeanVarianceType> meanTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(meanTensor,
                                                     5,
                                                     static_cast<MeanVarianceType>(0.0f),
                                                     static_cast<MeanVarianceType>(1.0f),
                                                     seed));

    PinnedTensor<MeanVarianceType> varianceTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(varianceTensor,
                                                     6,
                                                     static_cast<MeanVarianceType>(0.1f),
                                                     static_cast<MeanVarianceType>(1.0f),
                                                     seed));

    auto batchnormBuilder = TestCpuReferenceGraphExecutor::createValidBatchnormGraph(
        xTensor.strides(),
        xTensor.dims(),
        true,
        hipdnn_sdk::data_objects::DataType_FLOAT,
        hipdnn_sdk::data_objects::DataType_HALF,
        hipdnn_sdk::data_objects::DataType_HALF);

    auto batchnormGraph = batchnormBuilder.GetBufferPointer();

    std::unordered_map<int64_t, void*> variantPack;
    for(const auto& deviceBuffer : deviceBuffers)
    {
        variantPack[deviceBuffer.uid] = deviceBuffer.ptr;
    }

    hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor::execute(
        batchnormGraph, batchnormBuilder.GetSize(), variantPack);
}
