// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceImplementation.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::reference_test_utilities;

namespace
{

struct Batchnorm2dTestCase
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm2dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, h, w};
    }
};

template <typename InputType, typename IntermediateType>
struct Batchnorm2dTensorBundle
{
    Batchnorm2dTensorBundle(const std::vector<int64_t>& dims,
                            unsigned int seed = 1,
                            const TensorLayout& layout = TensorLayout::NCHW)
        : derivedDims({1, dims[1], 1, 1})
        , xTensor(dims, layout)
        , yTensor(dims, layout)
        , scaleTensor(derivedDims)
        , biasTensor(derivedDims)
        , meanTensor(derivedDims)
        , varianceTensor(derivedDims)
    {
        xTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        yTensor.fillWithRandomValues(
            static_cast<InputType>(-100.0f), static_cast<InputType>(100.0f), seed);

        scaleTensor.fillWithRandomValues(
            static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

        biasTensor.fillWithRandomValues(
            static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

        meanTensor.fillWithRandomValues(
            static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

        varianceTensor.fillWithRandomValues(
            static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f), seed);
    }

    std::vector<int64_t> derivedDims;
    PinnedTensor<InputType> xTensor;
    PinnedTensor<InputType> yTensor;
    PinnedTensor<IntermediateType> scaleTensor;
    PinnedTensor<IntermediateType> biasTensor;
    PinnedTensor<IntermediateType> meanTensor;
    PinnedTensor<IntermediateType> varianceTensor;
};

} // namespace

class BatchnormForwardInferenceIntegrationTest
    : public ::testing::TestWithParam<Batchnorm2dTestCase>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Uncomment if you want debug logging info.
        // setenv("HIPDNN_LOG_LEVEL", "info", 1);

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        // Note: The plugin paths has to be set before we create the hipdnn handle.
        const std::array<const char*, 1> paths = {PLUGIN_DIR};
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

    template <typename InputType, typename IntermediateType>
    std::unordered_map<int64_t, void*>
        createVariantPack(const graph::TensorAttributes& xTensorAttr,
                          const graph::TensorAttributes& yTensorAttr,
                          const graph::TensorAttributes& meanTensorAttr,
                          const graph::TensorAttributes& invVarianceTensorAttr,
                          const graph::TensorAttributes& scaleTensorAttr,
                          const graph::TensorAttributes& biasTensorAttr,
                          Batchnorm2dTensorBundle<InputType, IntermediateType>& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[meanTensorAttr.get_uid()] = tensorBundle.meanTensor.memory().deviceData();
        variantPack[invVarianceTensorAttr.get_uid()]
            = tensorBundle.varianceTensor.memory().deviceData();
        variantPack[scaleTensorAttr.get_uid()] = tensorBundle.scaleTensor.memory().deviceData();
        variantPack[biasTensorAttr.get_uid()] = tensorBundle.biasTensor.memory().deviceData();
        variantPack[yTensorAttr.get_uid()] = tensorBundle.yTensor.memory().deviceData();

        return variantPack;
    }

    template <typename InputType, typename IntermediateType>
    void runMiopenBatchnormFwd(
        Batchnorm2dTensorBundle<InputType, IntermediateType>& graphTensorBundle,
        DataType_t inputDataType,
        DataType_t intermediateDataType)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();

        graph->set_name("BatchnormInferenceTest");

        int64_t uid = 1;
        auto xAttr = graph::makeTensorAttributes("X", inputDataType, graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, graphTensorBundle.meanTensor);
        meanAttr.set_uid(uid++);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = graph::makeTensorAttributes(
            "inv_variance", intermediateDataType, graphTensorBundle.varianceTensor);
        invVarianceAttr.set_uid(uid++);
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, graphTensorBundle.scaleTensor);
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, graphTensorBundle.biasTensor);
        biasAttr.set_uid(uid++);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::BatchnormInferenceAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_inference");

        auto yTensorAttr = graph->batchnorm_inference(xTensorAttr,
                                                      meanTensorAttr,
                                                      invVarianceTensorAttr,
                                                      scaleTensorAttr,
                                                      biasTensorAttr,
                                                      bnAttrs);

        if(!yTensorAttr->has_uid())
        {
            HIPDNN_LOG_INFO("yTensorAttr does not have a UID, giving it a UID");
            yTensorAttr->set_uid(uid++);
        }

        yTensorAttr->set_data_type(inputDataType);

        // Validate and build graph
        auto result = graph->validate();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->create_execution_plans(_handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        auto variantPack = createVariantPack<InputType, IntermediateType>(*xTensorAttr,
                                                                          *yTensorAttr,
                                                                          *meanTensorAttr,
                                                                          *invVarianceTensorAttr,
                                                                          *scaleTensorAttr,
                                                                          *biasTensorAttr,
                                                                          graphTensorBundle);

        result = graph->execute(_handle, variantPack, _stream);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    }

    template <typename InputType, typename IntermediateType>
    void runCpuBatchnormFwd(Batchnorm2dTensorBundle<InputType, IntermediateType>& cpuTensorBundle)
    {
        CpuFpReferenceImplementation<InputType, IntermediateType, IntermediateType> cpuRefImpl;
        cpuRefImpl.batchnormFwdInference(cpuTensorBundle.xTensor,
                                         cpuTensorBundle.scaleTensor,
                                         cpuTensorBundle.biasTensor,
                                         cpuTensorBundle.meanTensor,
                                         cpuTensorBundle.varianceTensor,
                                         cpuTensorBundle.yTensor,
                                         1e-3);
    }

    template <typename InputType, typename IntermediateType>
    void runBatchnormTest(const Batchnorm2dTestCase& testCase,
                          InputType tolerance = 1e-4f,
                          const TensorLayout& layout = TensorLayout::NCHW)
    {
        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        unsigned int seed = std::random_device{}();
        //log the random seed in case we need to reproduce the test
        HIPDNN_LOG_INFO("Test is using {} for its random seed", seed);

        Batchnorm2dTensorBundle<InputType, IntermediateType> graphTensorBundle(
            testCase.getDims(), seed, layout);

        Batchnorm2dTensorBundle<InputType, IntermediateType> cpuTensorBundle(
            testCase.getDims(), seed, layout);

        runMiopenBatchnormFwd<InputType, IntermediateType>(
            graphTensorBundle, inputDataType, intermediateDataType);
        graphTensorBundle.yTensor.memory().markDeviceModified();

        runCpuBatchnormFwd<InputType, IntermediateType>(cpuTensorBundle);

        CpuFpReferenceValidation<InputType> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(cpuRefValidation.allClose(cpuTensorBundle.yTensor.memory(),
                                              graphTensorBundle.yTensor.memory()));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

namespace
{

std::vector<Batchnorm2dTestCase> getBnFwdInferenceTestCases()
{
    return {
        {.n = 1, .c = 3, .h = 14, .w = 14},
        {.n = 1, .c = 256, .h = 1, .w = 1},
        {.n = 2, .c = 3, .h = 1, .w = 1},
        {.n = 32, .c = 1, .h = 14, .w = 14},
        {.n = 32, .c = 3, .h = 1, .w = 14},
        {.n = 32, .c = 3, .h = 14, .w = 1},
        {.n = 64, .c = 64, .h = 112, .w = 112},
        {.n = 64, .c = 512, .h = 14, .w = 14},
    };
}

} // namespace

TEST_P(BatchnormForwardInferenceIntegrationTest, RunFloatFwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<float, float>(testCase, 1e-6f);
}

INSTANTIATE_TEST_SUITE_P(RunFloatFwdBatchnormGraph,
                         BatchnormForwardInferenceIntegrationTest,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

class BatchnormForwardInferenceIntegrationTestBfloat16
    : public BatchnormForwardInferenceIntegrationTest
{
};

TEST_P(BatchnormForwardInferenceIntegrationTestBfloat16, RunBfloat16FwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<hip_bfloat16, float>(testCase, 1e-2_bf);
}

INSTANTIATE_TEST_SUITE_P(RunBfloat16FwdBatchnormGraph,
                         BatchnormForwardInferenceIntegrationTestBfloat16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

class BatchnormForwardInferenceIntegrationTestHalf : public BatchnormForwardInferenceIntegrationTest
{
};
TEST_P(BatchnormForwardInferenceIntegrationTestHalf, RunHalfFwdbatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<half, float>(testCase, 1e-2_h);
}

INSTANTIATE_TEST_SUITE_P(RunHalfFwdbatchnormGraph,
                         BatchnormForwardInferenceIntegrationTestHalf,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

// Basic NHWC float test case
class BatchnormForwardInferenceIntegrationTestNhwc : public BatchnormForwardInferenceIntegrationTest
{
};

TEST_P(BatchnormForwardInferenceIntegrationTestNhwc, RunFloatFwdBatchnormGraphNHWC)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<float, float>(testCase, 1e-6f, TensorLayout::NHWC);
}

// Consider using fewer/smaller test cases to reduce test time
INSTANTIATE_TEST_SUITE_P(RunFloatFwdBatchnormGraphNHWC,
                         BatchnormForwardInferenceIntegrationTestNhwc,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

class BatchnormForwardInferenceIntegrationTestBfloat16Nhwc
    : public BatchnormForwardInferenceIntegrationTest
{
};

TEST_P(BatchnormForwardInferenceIntegrationTestBfloat16Nhwc, RunBfloat16FwdBatchnormGraphNHWC)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<hip_bfloat16, float>(testCase, 1e-2_bf, TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(RunBfloat16FwdBatchnormGraphNHWC,
                         BatchnormForwardInferenceIntegrationTestBfloat16Nhwc,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

class BatchnormForwardInferenceIntegrationTestHalfNhwc
    : public BatchnormForwardInferenceIntegrationTest
{
};

TEST_P(BatchnormForwardInferenceIntegrationTestHalfNhwc, RunHalfFwdBatchnormGraphNHWC)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<half, float>(testCase, 1e-2_h, TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(RunHalfFwdBatchnormGraphNHWC,
                         BatchnormForwardInferenceIntegrationTestHalfNhwc,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));
