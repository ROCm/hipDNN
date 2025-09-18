// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <hipdnn_sdk/utilities/Workspace.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/TestOperationsCommon.hpp"

using namespace hipdnn_sdk::test_utilities;
using namespace test_operations_common;

template <typename DataType>
class ConvFwdExecuteGraphBase : public ::testing::TestWithParam<ConvTestCase>
{
protected:
    TensorLayout _layout;

    ConvFwdExecuteGraphBase(TensorLayout layout)
        : _layout(std::move(layout))
    {
    }

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

    void runConvFwdGraph(const ConvTestCase& testCase,
                         hipdnn_sdk::data_objects::DataType inputDataType,
                         DataType epsilon)
    {
        unsigned int seed = std::random_device{}();

        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        PinnedTensor<DataType> xTensor(testCase._xDims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(
            xTensor, 1, static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed));

        PinnedTensor<DataType> wTensor(testCase._wDims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(
            wTensor, 2, static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed));

        PinnedTensor<DataType> yTensor(testCase._yDims, _layout);
        deviceBuffers.push_back(generateEmptyDeviceBuffer(yTensor, 3));

        auto convBuilder
            = hipdnn_backend::test_utilities::createValidConvFwdGraph(xTensor.dims(),
                                                                      xTensor.strides(),
                                                                      wTensor.dims(),
                                                                      wTensor.strides(),
                                                                      yTensor.dims(),
                                                                      yTensor.strides(),
                                                                      testCase._convPrePadding,
                                                                      testCase._convPostPadding,
                                                                      testCase._convStride,
                                                                      testCase._convDilation,
                                                                      inputDataType);

        hipdnnPluginConstData_t opGraph;
        opGraph.ptr = convBuilder.GetBufferPointer();
        opGraph.size = convBuilder.GetSize();

        auto engineConfigBuilder = hipdnn_backend::test_utilities::createValidEngineConfig(1);
        hipdnnPluginConstData_t engineConfig;
        engineConfig.ptr = engineConfigBuilder.GetBufferPointer();
        engineConfig.size = engineConfigBuilder.GetSize();

        hipdnnPluginStatus_t status;
        size_t workspaceSize;
        status
            = hipdnnEnginePluginGetWorkspaceSize(_handle, &engineConfig, &opGraph, &workspaceSize);
        EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
        hipdnn_sdk::utilities::Workspace workspace(workspaceSize);

        hipdnnEnginePluginExecutionContext_t executionContext;
        status = hipdnnEnginePluginCreateExecutionContext(
            _handle, &engineConfig, &opGraph, &executionContext);
        EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        status = hipdnnEnginePluginExecuteOpGraph(_handle,
                                                  executionContext,
                                                  workspace.get(),
                                                  deviceBuffers.data(),
                                                  static_cast<uint32_t>(deviceBuffers.size()));
        EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        yTensor.memory().markDeviceModified();

        status = hipdnnEnginePluginDestroyExecutionContext(_handle, executionContext);
        EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        Tensor<DataType> xTensorCpu(xTensor.dims(), _layout);
        xTensorCpu.fillWithRandomValues(
            static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed);
        Tensor<DataType> wTensorCpu(wTensor.dims(), _layout);
        wTensorCpu.fillWithRandomValues(
            static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), seed);
        Tensor<DataType> yTensorCpu(yTensor.dims(), _layout);

        CpuFpReferenceConvolutionImpl<DataType, float>::convFwdInference(xTensorCpu,
                                                                         wTensorCpu,
                                                                         yTensorCpu,
                                                                         testCase._convStride,
                                                                         testCase._convDilation,
                                                                         testCase._convPrePadding);

        CpuFpReferenceValidation<DataType> cpuRefValidationInput(epsilon, epsilon);

        EXPECT_TRUE(cpuRefValidationInput.allClose(yTensorCpu.memory(), yTensor.memory()));
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

class TestGpuMiopenConvFwdExecuteGraphNchwFp32 : public ConvFwdExecuteGraphBase<float>
{
public:
    TestGpuMiopenConvFwdExecuteGraphNchwFp32()
        : ConvFwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenConvFwdExecuteGraphNchwFp16 : public ConvFwdExecuteGraphBase<half>
{
public:
    TestGpuMiopenConvFwdExecuteGraphNchwFp16()
        : ConvFwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenConvFwdExecuteGraphNchwBfp16 : public ConvFwdExecuteGraphBase<hip_bfloat16>
{
public:
    TestGpuMiopenConvFwdExecuteGraphNchwBfp16()
        : ConvFwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenConvFwdExecuteGraphNhwcFp32 : public ConvFwdExecuteGraphBase<float>
{
public:
    TestGpuMiopenConvFwdExecuteGraphNhwcFp32()
        : ConvFwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenConvFwdExecuteGraphNhwcFp16 : public ConvFwdExecuteGraphBase<half>
{
public:
    TestGpuMiopenConvFwdExecuteGraphNhwcFp16()
        : ConvFwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenConvFwdExecuteGraphNhwcBfp16 : public ConvFwdExecuteGraphBase<hip_bfloat16>
{
public:
    TestGpuMiopenConvFwdExecuteGraphNhwcBfp16()
        : ConvFwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

TEST_P(TestGpuMiopenConvFwdExecuteGraphNchwFp32, Correctness)
{
    const ConvTestCase& testCase = GetParam();
    runConvFwdGraph(testCase, hipdnn_sdk::data_objects::DataType::FLOAT, 1e-6f);
}

TEST_P(TestGpuMiopenConvFwdExecuteGraphNchwBfp16, Correctness)
{
    const ConvTestCase& testCase = GetParam();
    runConvFwdGraph(testCase, hipdnn_sdk::data_objects::DataType::BFLOAT16, 1e-4_bf);
}

TEST_P(TestGpuMiopenConvFwdExecuteGraphNchwFp16, Correctness)
{
    const ConvTestCase& testCase = GetParam();
    runConvFwdGraph(testCase, hipdnn_sdk::data_objects::DataType::HALF, 1e-4_h);
}

TEST_P(TestGpuMiopenConvFwdExecuteGraphNhwcFp32, Correctness)
{
    const ConvTestCase& testCase = GetParam();
    runConvFwdGraph(testCase, hipdnn_sdk::data_objects::DataType::FLOAT, 1e-6f);
}

TEST_P(TestGpuMiopenConvFwdExecuteGraphNhwcBfp16, Correctness)
{
    const ConvTestCase& testCase = GetParam();
    runConvFwdGraph(testCase, hipdnn_sdk::data_objects::DataType::BFLOAT16, 1e-4_bf);
}

TEST_P(TestGpuMiopenConvFwdExecuteGraphNhwcFp16, Correctness)
{
    const ConvTestCase& testCase = GetParam();
    runConvFwdGraph(testCase, hipdnn_sdk::data_objects::DataType::HALF, 1e-4_h);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdExecuteGraphNchwFp32,
                         testing::ValuesIn(getConvTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdExecuteGraphNchwFp16,
                         testing::ValuesIn(getConvTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdExecuteGraphNchwBfp16,
                         testing::ValuesIn(getConvTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdExecuteGraphNhwcFp32,
                         testing::ValuesIn(getConvTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdExecuteGraphNhwcFp16,
                         testing::ValuesIn(getConvTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdExecuteGraphNhwcBfp16,
                         testing::ValuesIn(getConvTestCases()));
