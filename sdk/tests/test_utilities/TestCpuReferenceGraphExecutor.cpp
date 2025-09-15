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
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/TensorView.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

class TestCpuReferenceGraphExecutor
{
private:
public:
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
    using IntermediateType = float;
    TensorLayout layout = TensorLayout::NCHW;

    PinnedTensor<InputType> xTensor(dims, layout);
    deviceBuffers.push_back(generateRandomHostBuffer(
        xTensor, 1, static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed));

    PinnedTensor<InputType> yTensor(dims, layout);
    deviceBuffers.push_back(generateEmptyHostBuffer(yTensor, 2));

    PinnedTensor<IntermediateType> scaleTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(scaleTensor,
                                                     3,
                                                     static_cast<IntermediateType>(0.0f),
                                                     static_cast<IntermediateType>(1.0f),
                                                     seed));

    PinnedTensor<IntermediateType> biasTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(biasTensor,
                                                     4,
                                                     static_cast<IntermediateType>(0.0f),
                                                     static_cast<IntermediateType>(1.0f),
                                                     seed));

    PinnedTensor<IntermediateType> meanTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(meanTensor,
                                                     5,
                                                     static_cast<IntermediateType>(0.0f),
                                                     static_cast<IntermediateType>(1.0f),
                                                     seed));

    PinnedTensor<IntermediateType> varianceTensor(derivedDims);
    deviceBuffers.push_back(generateRandomHostBuffer(varianceTensor,
                                                     6,
                                                     static_cast<IntermediateType>(0.1f),
                                                     static_cast<IntermediateType>(1.0f),
                                                     seed));

    auto batchnormBuilder = hipdnn_backend::test_utilities::createValidBatchnormGraph(
        xTensor.strides(), xTensor.dims(), true, hipdnn_sdk::data_objects::DataType_FLOAT);

    auto batchnormGraph = batchnormBuilder.GetBufferPointer();

    std::unordered_map<int64_t, void*> variantPack;
    for(const auto& deviceBuffer : deviceBuffers)
    {
        variantPack[deviceBuffer.uid] = deviceBuffer.ptr;
    }

    hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor::executeTheGraph(
        batchnormGraph, batchnormBuilder.GetSize(), variantPack);
}
