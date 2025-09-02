// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceImplementation.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/HalfUtils.hpp>
#include <hipdnn_sdk/utilities/HipBfloat16Utils.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/TestOperationsCommon.hpp"

using namespace hipdnn_sdk::reference_test_utilities;
using namespace test_operations_common;

class BatchnormBwdExecuteGraphTest : public ::testing::TestWithParam<Batchnorm2dTestCase>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        hipdnnPluginStatus_t status = hipdnnEnginePluginCreate(&_handle);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnEnginePluginDestroy(_handle);
        }
    }

    template <typename InputType, typename IntermediateType>
    void runBwdBatchnormGraph(Batchnorm2dTestCase testCase,
                              hipdnn_sdk::data_objects::DataType inputDataType,
                              InputType epsilon,
                              const TensorLayout& layout);

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

TEST_P(BatchnormBwdExecuteGraphTest, RunFloatBwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<float, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 4e-3f, TensorLayout::NCHW);
}

TEST_P(BatchnormBwdExecuteGraphTest, RunBfloat16BwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<hip_bfloat16, float>(testCase,
                                              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
                                              4e-3_bf,
                                              TensorLayout::NCHW);
}

TEST_P(BatchnormBwdExecuteGraphTest, RunHalfBwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<half, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 4e-3_h, TensorLayout::NCHW);
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(BatchnormBwdExecuteGraphTest, DISABLED_RunDoubleBwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<double, double>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 4e-3, TensorLayout::NCHW);
}

TEST_P(BatchnormBwdExecuteGraphTest, RunFloatBwdBatchnormGraphNHWC)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<float, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 4e-3f, TensorLayout::NHWC);
}

// TODO: add unique test suite and conform to naming rules

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197
TEST_P(BatchnormBwdExecuteGraphTest, DISABLED_RunBfloat16BwdBatchnormGraphNHWC)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<hip_bfloat16, float>(testCase,
                                              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
                                              4e-3_bf,
                                              TensorLayout::NHWC);
}

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197
TEST_P(BatchnormBwdExecuteGraphTest, DISABLED_RunHalfBwdBatchnormGraphNHWC)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<half, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 4e-3_h, TensorLayout::NHWC);
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(BatchnormBwdExecuteGraphTest, DISABLED_RunDoubleBwdBatchnormGraphNHWC)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<double, double>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 4e-3, TensorLayout::NHWC);
}

template <typename InputType, typename IntermediateType>
void BatchnormBwdExecuteGraphTest::runBwdBatchnormGraph(
    Batchnorm2dTestCase testCase,
    hipdnn_sdk::data_objects::DataType inputDataType,
    InputType epsilon,
    const TensorLayout& layout)
{
    unsigned int seed = std::random_device{}();

    std::vector<int64_t> dims = {testCase.n, testCase.c, testCase.h, testCase.w};

    std::vector<int64_t> derivedDims = {1, dims[1], 1, 1};

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

    PinnedTensor<InputType> xTensor(dims, layout);
    deviceBuffers.push_back(generateRandomDeviceBuffer(
        xTensor, 1, static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed));

    PinnedTensor<InputType> dyTensor(dims, layout);
    deviceBuffers.push_back(generateRandomDeviceBuffer(
        dyTensor, 2, static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed));

    PinnedTensor<InputType> dxTensor(dims, layout);
    deviceBuffers.push_back(generateEmptyDeviceBuffer(dxTensor, 3));

    PinnedTensor<IntermediateType> scaleTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(scaleTensor,
                                                       4,
                                                       static_cast<IntermediateType>(-0.1f),
                                                       static_cast<IntermediateType>(0.1f),
                                                       seed));

    PinnedTensor<IntermediateType> dscaleTensor(derivedDims);
    deviceBuffers.push_back(generateEmptyDeviceBuffer(dscaleTensor, 5));

    PinnedTensor<IntermediateType> dbiasTensor(derivedDims);
    deviceBuffers.push_back(generateEmptyDeviceBuffer(dbiasTensor, 6));

    PinnedTensor<IntermediateType> meanTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(meanTensor,
                                                       7,
                                                       static_cast<IntermediateType>(-0.1f),
                                                       static_cast<IntermediateType>(0.1f),
                                                       seed));

    PinnedTensor<IntermediateType> invVarianceTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(invVarianceTensor,
                                                       8,
                                                       static_cast<IntermediateType>(1.9f),
                                                       static_cast<IntermediateType>(2.0f),
                                                       seed));

    auto batchnormBuilder = flatbuffer_test_utils::createValidBatchnormBwdGraph(
        dyTensor.strides(), dyTensor.dims(), true, inputDataType);

    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = batchnormBuilder.GetBufferPointer();
    opGraph.size = batchnormBuilder.GetSize();

    auto engineConfigBuilder = flatbuffer_test_utils::createValidEngineConfig(1);
    hipdnnPluginConstData_t engineConfig;
    engineConfig.ptr = engineConfigBuilder.GetBufferPointer();
    engineConfig.size = engineConfigBuilder.GetSize();

    hipdnnEnginePluginExecutionContext_t executionContext;
    hipdnnEnginePluginCreateExecutionContext(_handle, &engineConfig, &opGraph, &executionContext);

    hipdnnPluginStatus_t status
        = hipdnnEnginePluginExecuteOpGraph(_handle,
                                           executionContext,
                                           nullptr,
                                           deviceBuffers.data(),
                                           static_cast<uint32_t>(deviceBuffers.size()));
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

    dxTensor.memory().markDeviceModified();
    dscaleTensor.memory().markDeviceModified();
    dbiasTensor.memory().markDeviceModified();

    hipdnnEnginePluginDestroyExecutionContext(_handle, executionContext);

    Tensor<InputType> xTensorCpu(dims, layout);
    xTensorCpu.fillWithRandomValues(
        static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed);
    Tensor<InputType> dyTensorCpu(dims, layout);
    dyTensorCpu.fillWithRandomValues(
        static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed);
    Tensor<InputType> dxTensorCpu(dims, layout);

    Tensor<IntermediateType> scaleTensorCpu(derivedDims);
    scaleTensorCpu.fillWithRandomValues(
        static_cast<IntermediateType>(-0.1f), static_cast<IntermediateType>(0.1f), seed);
    Tensor<IntermediateType> dscaleTensorCpu(derivedDims);
    Tensor<IntermediateType> dbiasTensorCpu(derivedDims);
    Tensor<IntermediateType> meanTensorCpu(derivedDims);
    meanTensorCpu.fillWithRandomValues(
        static_cast<IntermediateType>(-0.1f), static_cast<IntermediateType>(0.1f), seed);

    Tensor<IntermediateType> invVarianceTensorCpu(derivedDims);
    invVarianceTensorCpu.fillWithRandomValues(
        static_cast<IntermediateType>(1.9f), static_cast<IntermediateType>(2.0f), seed);

    CpuFpReferenceImplementation<InputType, IntermediateType, IntermediateType> cpuRefImpl;
    cpuRefImpl.batchnormBwd(dyTensorCpu,
                            xTensorCpu,
                            meanTensorCpu,
                            invVarianceTensorCpu,
                            scaleTensorCpu,
                            dxTensorCpu,
                            dscaleTensorCpu,
                            dbiasTensorCpu);

    CpuFpReferenceValidation<InputType> cpuRefValidationInput(epsilon, epsilon);
    CpuFpReferenceValidation<IntermediateType> cpuRefValidationIntermediate(epsilon, epsilon);

    EXPECT_TRUE(cpuRefValidationInput.allClose(dxTensorCpu.memory(), dxTensor.memory()));
    EXPECT_TRUE(
        cpuRefValidationIntermediate.allClose(dscaleTensorCpu.memory(), dscaleTensor.memory()));
    EXPECT_TRUE(
        cpuRefValidationIntermediate.allClose(dbiasTensorCpu.memory(), dbiasTensor.memory()));
}

INSTANTIATE_TEST_SUITE_P(RunBwdBatchnormGraphWithParams,
                         BatchnormBwdExecuteGraphTest,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
