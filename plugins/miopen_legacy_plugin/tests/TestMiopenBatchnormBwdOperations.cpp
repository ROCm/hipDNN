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

using namespace hipdnn_sdk::test_utilities;
using namespace test_operations_common;

template <typename InputType, typename IntermediateType>
class BatchnormBwdExecuteGraphBase : public ::testing::TestWithParam<Batchnorm2dTestCase>
{
protected:
    TensorLayout _layout;

    BatchnormBwdExecuteGraphBase(TensorLayout layout)
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

    void runBwdBatchnormGraph(Batchnorm2dTestCase testCase,
                              hipdnn_sdk::data_objects::DataType inputDataType,
                              InputType epsilon)
    {
        unsigned int seed = std::random_device{}();

        std::vector<int64_t> dims = {testCase.n, testCase.c, testCase.h, testCase.w};

        std::vector<int64_t> derivedDims = {1, dims[1]};

        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        PinnedTensor<InputType> xTensor(dims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(
            xTensor, 1, static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed));

        PinnedTensor<InputType> dyTensor(dims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(
            dyTensor, 2, static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed));

        PinnedTensor<InputType> dxTensor(dims, _layout);
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

        auto batchnormBuilder = hipdnn_backend::test_utilities::createValidBatchnormBwdGraph(
            dyTensor.strides(), dyTensor.dims(), true, inputDataType);

        hipdnnPluginConstData_t opGraph;
        opGraph.ptr = batchnormBuilder.GetBufferPointer();
        opGraph.size = batchnormBuilder.GetSize();

        auto engineConfigBuilder = hipdnn_backend::test_utilities::createValidEngineConfig(1);
        hipdnnPluginConstData_t engineConfig;
        engineConfig.ptr = engineConfigBuilder.GetBufferPointer();
        engineConfig.size = engineConfigBuilder.GetSize();

        hipdnnEnginePluginExecutionContext_t executionContext;
        hipdnnEnginePluginCreateExecutionContext(
            _handle, &engineConfig, &opGraph, &executionContext);

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

        Tensor<InputType> xTensorCpu(dims, _layout);
        xTensorCpu.fillWithRandomValues(
            static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed);
        Tensor<InputType> dyTensorCpu(dims, _layout);
        dyTensorCpu.fillWithRandomValues(
            static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed);
        Tensor<InputType> dxTensorCpu(dims, _layout);

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

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

class TestGpuMiopenBatchnormBwdExecuteGraphFp32Nchw
    : public BatchnormBwdExecuteGraphBase<float, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphFp32Nchw()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphFp16Nchw
    : public BatchnormBwdExecuteGraphBase<half, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphFp16Nchw()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphBfp16Nchw
    : public BatchnormBwdExecuteGraphBase<hip_bfloat16, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphBfp16Nchw()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphFp64Nchw
    : public BatchnormBwdExecuteGraphBase<double, double>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphFp64Nchw()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphFp32Nhwc
    : public BatchnormBwdExecuteGraphBase<float, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphFp32Nhwc()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphFp16Nhwc
    : public BatchnormBwdExecuteGraphBase<half, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphFp16Nhwc()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphBfp16Nhwc
    : public BatchnormBwdExecuteGraphBase<hip_bfloat16, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphBfp16Nhwc()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphFp64Nhwc
    : public BatchnormBwdExecuteGraphBase<double, double>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphFp64Nhwc()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphFp32Nchw, Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 4e-3f);
}

TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphBfp16Nchw, Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16, 4e-3_bf);
}

TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphFp16Nchw, Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 4e-3_h);
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphFp64Nchw, DISABLED_Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 4e-3);
}

TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphFp32Nhwc, Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 4e-3f);
}

// TODO: add unique test suite and conform to naming rules

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197
TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphBfp16Nhwc, DISABLED_Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16, 4e-3_bf);
}

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197
TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphFp16Nhwc, DISABLED_Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 4e-3_h);
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphFp64Nhwc, DISABLED_Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 4e-3);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphFp32Nchw,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphFp16Nchw,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphBfp16Nchw,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphFp64Nchw,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphFp32Nhwc,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphFp16Nhwc,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphBfp16Nhwc,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphFp64Nhwc,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
