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
class BatchnormFwdInferenceExecuteGraphBase : public ::testing::TestWithParam<Batchnorm2dTestCase>
{
protected:
    TensorLayout _layout;

    BatchnormFwdInferenceExecuteGraphBase(TensorLayout layout)
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

    void runFwdBatchnormGraph(Batchnorm2dTestCase testCase,
                              hipdnn_sdk::data_objects::DataType inputDataType,
                              InputType epsilon)
    {
        auto seed = std::random_device{}();

        auto dims = std::vector<int64_t>{testCase.n, testCase.c, testCase.h, testCase.w};

        auto derivedDims = std::vector<int64_t>{1, dims[1], 1, 1};

        auto deviceBuffers = std::vector<hipdnnPluginDeviceBuffer_t>{};

        PinnedTensor<InputType> xTensor(dims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(
            xTensor, 1, static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed));

        PinnedTensor<InputType> yTensor(dims, _layout);
        deviceBuffers.push_back(generateEmptyDeviceBuffer(yTensor, 2));

        PinnedTensor<IntermediateType> scaleTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(scaleTensor,
                                                           3,
                                                           static_cast<IntermediateType>(0.0f),
                                                           static_cast<IntermediateType>(1.0f),
                                                           seed));

        PinnedTensor<IntermediateType> biasTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(biasTensor,
                                                           4,
                                                           static_cast<IntermediateType>(0.0f),
                                                           static_cast<IntermediateType>(1.0f),
                                                           seed));

        PinnedTensor<IntermediateType> meanTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(meanTensor,
                                                           5,
                                                           static_cast<IntermediateType>(0.0f),
                                                           static_cast<IntermediateType>(1.0f),
                                                           seed));

        PinnedTensor<IntermediateType> varianceTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(varianceTensor,
                                                           6,
                                                           static_cast<IntermediateType>(0.1f),
                                                           static_cast<IntermediateType>(1.0f),
                                                           seed));

        auto batchnormBuilder = hipdnn_backend::test_utilities::createValidBatchnormGraph(
            xTensor.strides(), xTensor.dims(), true, inputDataType);

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

        yTensor.memory().markDeviceModified();

        hipdnnEnginePluginDestroyExecutionContext(_handle, executionContext);

        Tensor<InputType> xTensorCpu(dims, _layout);
        xTensorCpu.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        Tensor<InputType> yTensorCpu(dims, _layout);
        Tensor<IntermediateType> scaleTensorCpu(derivedDims);
        scaleTensorCpu.fillWithRandomValues(
            static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);
        Tensor<IntermediateType> biasTensorCpu(derivedDims);
        biasTensorCpu.fillWithRandomValues(
            static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);
        Tensor<IntermediateType> meanTensorCpu(derivedDims);
        meanTensorCpu.fillWithRandomValues(
            static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);
        Tensor<IntermediateType> varianceTensorCpu(derivedDims);
        varianceTensorCpu.fillWithRandomValues(
            static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f), seed);

        CpuFpReferenceImplementation<InputType, IntermediateType, IntermediateType> cpuRefImpl;
        cpuRefImpl.batchnormFwdInference(xTensorCpu,
                                         scaleTensorCpu,
                                         biasTensorCpu,
                                         meanTensorCpu,
                                         varianceTensorCpu,
                                         yTensorCpu,
                                         1e-3);

        CpuFpReferenceValidation<InputType> cpuRefValidation(epsilon, epsilon);
        EXPECT_TRUE(cpuRefValidation.allClose(yTensorCpu.memory(), yTensor.memory()));
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp32Nchw
    : public BatchnormFwdInferenceExecuteGraphBase<float, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp32Nchw()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp16Nchw
    : public BatchnormFwdInferenceExecuteGraphBase<half, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp16Nchw()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphBfp16Nchw
    : public BatchnormFwdInferenceExecuteGraphBase<hip_bfloat16, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphBfp16Nchw()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp64Nchw
    : public BatchnormFwdInferenceExecuteGraphBase<double, double>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp64Nchw()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp32Nhwc
    : public BatchnormFwdInferenceExecuteGraphBase<float, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp32Nhwc()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp16Nhwc
    : public BatchnormFwdInferenceExecuteGraphBase<half, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp16Nhwc()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphBfp16Nhwc
    : public BatchnormFwdInferenceExecuteGraphBase<hip_bfloat16, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphBfp16Nhwc()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp64Nhwc
    : public BatchnormFwdInferenceExecuteGraphBase<double, double>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp64Nhwc()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp32Nchw, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 1e-6f);
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphBfp16Nchw, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16, 1e-2_bf);
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp16Nchw, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 1e-2_h);
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp64Nchw, DISABLED_Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 1e-6);
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp32Nhwc, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 1e-6f);
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphBfp16Nhwc, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16, 1e-2_bf);
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp16Nhwc, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 1e-2_h);
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp64Nhwc, DISABLED_Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp32Nchw,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp16Nchw,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphBfp16Nchw,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp64Nchw,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp32Nhwc,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp16Nhwc,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphBfp16Nhwc,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphFp64Nhwc,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
