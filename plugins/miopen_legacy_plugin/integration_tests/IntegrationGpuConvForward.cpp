// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <random>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/StringUtil.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/Workspace.hpp>

#include "../tests/common/ConvolutionCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace test_conv_common;

namespace
{

template <typename DataType>
class ConvForward : public ::testing::TestWithParam<ConvTestCase>
{
    struct ConvTensorBundle
    {
        ConvTensorBundle(const ConvTestCase& testCase,
                         const TensorLayout& layout = TensorLayout::NCHW)
            : xTensor(testCase._xDims, layout)
            , wTensor(testCase._wDims, layout)
            , yTensor(testCase._yDims, layout)
        {
            xTensor.fillWithRandomValues(
                static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase._seed);
            wTensor.fillWithRandomValues(
                static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase._seed);
            yTensor.fillWithValue(static_cast<DataType>(0.0));
        }

        PinnedTensor<DataType> xTensor;
        PinnedTensor<DataType> wTensor;
        PinnedTensor<DataType> yTensor;
    };

protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        // Note: The plugin paths has to be set before we create the hipdnn handle.
        const std::array<const char*, 1> paths = {PLUGIN_PATH};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        // Create handle and stream
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    std::unordered_map<int64_t, void*> createVariantPack(const graph::TensorAttributes& xTensorAttr,
                                                         const graph::TensorAttributes& wTensorAttr,
                                                         const graph::TensorAttributes& yTensorAttr,
                                                         ConvTensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[wTensorAttr.get_uid()] = tensorBundle.wTensor.memory().deviceData();
        variantPack[yTensorAttr.get_uid()] = tensorBundle.yTensor.memory().deviceData();

        return variantPack;
    }

    void runMiopenConvFwd(const ConvTestCase& testCase,
                          ConvTensorBundle& graphTensorBundle,
                          hipdnn_frontend::DataType inputDataType)
    {
        auto graphObj = std::make_shared<hipdnn_frontend::graph::Graph>();

        graphObj->set_name("ConvolutionForwardTest");

        int64_t uid = 1;

        auto xAttr = graph::makeTensorAttributes("x", inputDataType, graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr = graph::makeTensorAttributes("w", inputDataType, graphTensorBundle.wTensor);
        wAttr.set_uid(uid++);
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_name("convolution_forward");
        convAttrs.set_pre_padding(testCase._convPrePadding);
        convAttrs.set_post_padding(testCase._convPostPadding);
        convAttrs.set_stride(testCase._convStride);
        convAttrs.set_dilation(testCase._convDilation);

        auto yTensorAttr = graphObj->conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

        if(!yTensorAttr->has_uid())
        {
            yTensorAttr->set_uid(uid++);
        }
        yTensorAttr->set_data_type(inputDataType);

        auto result = graphObj->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graphObj->build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graphObj->create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graphObj->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graphObj->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t workspaceSize;
        result = graphObj->get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0) << result.err_msg;
        Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack
            = createVariantPack(*xTensorAttr, *wTensorAttr, *yTensorAttr, graphTensorBundle);

        result = graphObj->execute(_handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    void runCpuConvFwd(const ConvTestCase& testCase, ConvTensorBundle& cpuTensorBundle)
    {
        CpuFpReferenceConvolutionImpl<DataType, float>::convFwdInference(cpuTensorBundle.xTensor,
                                                                         cpuTensorBundle.wTensor,
                                                                         cpuTensorBundle.yTensor,
                                                                         testCase._convStride,
                                                                         testCase._convDilation,
                                                                         testCase._convPrePadding);
    }

    void runConvTest(DataType tolerance = 1e-4f, const TensorLayout& layout = TensorLayout::NCHW)
    {
        const ConvTestCase& testCase = GetParam();

        auto inputDataType = getDataTypeEnumFromType<DataType>();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", testCase._seed);

        ConvTensorBundle graphTensorBundle(testCase, layout);

        ConvTensorBundle cpuTensorBundle(testCase, layout);

        runMiopenConvFwd(testCase, graphTensorBundle, inputDataType);
        graphTensorBundle.yTensor.memory().markDeviceModified();

        runCpuConvFwd(testCase, cpuTensorBundle);

        CpuFpReferenceValidation<DataType> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(cpuRefValidation.allClose(cpuTensorBundle.yTensor.memory(),
                                              graphTensorBundle.yTensor.memory()));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

class IntegrationGpuConvFwdNchwFp32 : public ConvForward<float>
{
};

class IntegrationGpuConvFwdNchwBfp16 : public ConvForward<hip_bfloat16>
{
};

class IntegrationGpuConvFwdNchwFp16 : public ConvForward<half>
{
};

class IntegrationGpuConvFwdNhwcFp32 : public ConvForward<float>
{
};

class IntegrationGpuConvFwdNhwcBfp16 : public ConvForward<hip_bfloat16>
{
};

class IntegrationGpuConvFwdNhwcFp16 : public ConvForward<half>
{
};

std::vector<ConvTestCase> getConvFwdTestCases()
{
    return getConvTestCases();
}

} // namespace

TEST_P(IntegrationGpuConvFwdNchwFp32, Correctness)
{
    runConvTest(4e-6f, TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNchwFp32, testing::ValuesIn(getConvFwdTestCases()));

TEST_P(IntegrationGpuConvFwdNchwBfp16, Correctness)
{
    runConvTest(1e-2_bf, TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvFwdNchwBfp16,
                         testing::ValuesIn(getConvFwdTestCases()));

TEST_P(IntegrationGpuConvFwdNchwFp16, Correctness)
{
    runConvTest(1e-2_h, TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNchwFp16, testing::ValuesIn(getConvFwdTestCases()));

TEST_P(IntegrationGpuConvFwdNhwcFp32, Correctness)
{
    runConvTest(4e-6f, TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNhwcFp32, testing::ValuesIn(getConvFwdTestCases()));

TEST_P(IntegrationGpuConvFwdNhwcBfp16, Correctness)
{
    runConvTest(1e-2_bf, TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvFwdNhwcBfp16,
                         testing::ValuesIn(getConvFwdTestCases()));

TEST_P(IntegrationGpuConvFwdNhwcFp16, Correctness)
{
    runConvTest(1e-3_h, TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNhwcFp16, testing::ValuesIn(getConvFwdTestCases()));
