// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_frontend/attributes/tensor_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_frontend/utilities.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

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
                            const Tensor_layout& layout = Tensor_layout::NCHW)
        : derivedDims({1, dims[1], 1, 1})
        , xTensor(dims, layout)
        , dyTensor(dims, layout)
        , dxTensor(dims, layout)
        , scaleTensor(derivedDims)
        , dscaleTensor(derivedDims)
        , dbiasTensor(derivedDims)
        , meanTensor(derivedDims)
        , invVarianceTensor(derivedDims)
    {
        xTensor.fill_with_random_values(
            static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed);

        dyTensor.fill_with_random_values(
            static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed);
        scaleTensor.fill_with_random_values(
            static_cast<IntermediateType>(-0.1f), static_cast<IntermediateType>(0.1f), seed);

        meanTensor.fill_with_random_values(
            static_cast<IntermediateType>(-0.1f), static_cast<IntermediateType>(0.1f), seed);

        invVarianceTensor.fill_with_random_values(
            static_cast<IntermediateType>(1.9f), static_cast<IntermediateType>(2.0f), seed);
    }

    std::vector<int64_t> derivedDims;
    PinnedTensor<InputType> xTensor;
    PinnedTensor<InputType> dyTensor;
    PinnedTensor<InputType> dxTensor;
    PinnedTensor<IntermediateType> scaleTensor;
    PinnedTensor<IntermediateType> dscaleTensor;
    PinnedTensor<IntermediateType> dbiasTensor;
    PinnedTensor<IntermediateType> meanTensor;
    PinnedTensor<IntermediateType> invVarianceTensor;
};

} // namespace

class BatchnormBackwardIntegrationTest : public ::testing::TestWithParam<Batchnorm2dTestCase>
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

        const std::array<const char*, 1> paths = {PLUGIN_DIR};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

        //todo: bring back stream support once MigratableMemory supports it
        //ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        //ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);
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
                          const graph::TensorAttributes& dyTensorAttr,
                          const graph::TensorAttributes& dxTensorAttr,
                          const graph::TensorAttributes& scaleTensorAttr,
                          const graph::TensorAttributes& dscaleTensorAttr,
                          const graph::TensorAttributes& dbiasTensorAttr,
                          const graph::TensorAttributes& meanTensorAttr,
                          const graph::TensorAttributes& invVarianceTensorAttr,
                          Batchnorm2dTensorBundle<InputType, IntermediateType>& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = tensorBundle.xTensor.memory().device_data();
        variantPack[dyTensorAttr.get_uid()] = tensorBundle.dyTensor.memory().device_data();
        variantPack[dxTensorAttr.get_uid()] = tensorBundle.dxTensor.memory().device_data();
        variantPack[scaleTensorAttr.get_uid()] = tensorBundle.scaleTensor.memory().device_data();
        variantPack[dscaleTensorAttr.get_uid()] = tensorBundle.dscaleTensor.memory().device_data();
        variantPack[dbiasTensorAttr.get_uid()] = tensorBundle.dbiasTensor.memory().device_data();
        variantPack[meanTensorAttr.get_uid()] = tensorBundle.meanTensor.memory().device_data();
        variantPack[invVarianceTensorAttr.get_uid()]
            = tensorBundle.invVarianceTensor.memory().device_data();

        return variantPack;
    }

    template <typename InputType, typename IntermediateType>
    void runMiopenBatchnormBwd(
        Batchnorm2dTensorBundle<InputType, IntermediateType>& graphTensorBundle,
        DataType_t inputDataType,
        DataType_t intermediateDataType)
    {
        auto graphObj = std::make_shared<hipdnn_frontend::graph::Graph>();

        graphObj->set_name("BatchnormBackwardTest");

        int64_t uid = 1;

        auto xAttr = graph::make_tensor_attributes("x", inputDataType, graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr
            = graph::make_tensor_attributes("dy", inputDataType, graphTensorBundle.dyTensor);
        dyAttr.set_uid(uid++);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto scaleAttr = graph::make_tensor_attributes(
            "scale", intermediateDataType, graphTensorBundle.scaleTensor);
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto meanAttr = graph::make_tensor_attributes(
            "mean", intermediateDataType, graphTensorBundle.meanTensor);
        meanAttr.set_uid(uid++);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = graph::make_tensor_attributes(
            "inv_variance", intermediateDataType, graphTensorBundle.invVarianceTensor);
        invVarianceAttr.set_uid(uid++);
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        graph::BatchnormBackwardAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_backward");
        bnAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);

        auto outputTensorsAttr
            = graphObj->batchnorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, bnAttrs);

        auto& dxTensorAttr = outputTensorsAttr[0];
        if(!dxTensorAttr->has_uid())
        {
            HIPDNN_LOG_INFO("dxTensorAttr does not have a UID, giving it a UID");
            dxTensorAttr->set_uid(uid++);
        }
        dxTensorAttr->set_data_type(inputDataType);

        auto& dscaleTensorAttr = outputTensorsAttr[1];
        if(!dscaleTensorAttr->has_uid())
        {
            HIPDNN_LOG_INFO("dscaleTensorAttr does not have a UID, giving it a UID");
            dscaleTensorAttr->set_uid(uid++);
        }
        dscaleTensorAttr->set_data_type(intermediateDataType);

        auto& dbiasTensorAttr = outputTensorsAttr[2];
        if(!dbiasTensorAttr->has_uid())
        {
            HIPDNN_LOG_INFO("dbiasTensorAttr does not have a UID, giving it a UID");
            dbiasTensorAttr->set_uid(uid++);
        }
        dbiasTensorAttr->set_data_type(intermediateDataType);

        auto result = graphObj->validate();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graphObj->build_operation_graph(_handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graphObj->create_execution_plans(_handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graphObj->check_support();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graphObj->build_plans();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        auto variantPack = createVariantPack<InputType, IntermediateType>(*xTensorAttr,
                                                                          *dyTensorAttr,
                                                                          *dxTensorAttr,
                                                                          *scaleTensorAttr,
                                                                          *dscaleTensorAttr,
                                                                          *dbiasTensorAttr,
                                                                          *meanTensorAttr,
                                                                          *invVarianceTensorAttr,
                                                                          graphTensorBundle);

        result = graphObj->execute(_handle, variantPack, nullptr);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    }

    template <typename InputType, typename IntermediateType>
    void runCpuBatchnormBwd(Batchnorm2dTensorBundle<InputType, IntermediateType>& cpuTensorBundle)
    {
        Cpu_fp_reference_implementation<InputType, IntermediateType, IntermediateType> cpuRefImpl;
        cpuRefImpl.batchnorm_bwd(cpuTensorBundle.dyTensor,
                                 cpuTensorBundle.xTensor,
                                 cpuTensorBundle.meanTensor,
                                 cpuTensorBundle.invVarianceTensor,
                                 cpuTensorBundle.scaleTensor,
                                 cpuTensorBundle.dxTensor,
                                 cpuTensorBundle.dscaleTensor,
                                 cpuTensorBundle.dbiasTensor);
    }

    template <typename InputType, typename IntermediateType>
    void runBatchnormTest(const Batchnorm2dTestCase& testCase,
                          InputType tolerance = 1e4f,
                          const Tensor_layout& layout = Tensor_layout::NCHW)
    {
        auto inputDataType = get_data_type_enum_from_type<InputType>();
        auto intermediateDataType = get_data_type_enum_from_type<IntermediateType>();

        unsigned int seed = std::random_device{}();
        HIPDNN_LOG_INFO("Test is using {} for its random seed", seed);

        Batchnorm2dTensorBundle<InputType, IntermediateType> graphTensorBundle(
            testCase.getDims(), seed, layout);

        Batchnorm2dTensorBundle<InputType, IntermediateType> cpuTensorBundle(
            testCase.getDims(), seed, layout);

        runMiopenBatchnormBwd<InputType, IntermediateType>(
            graphTensorBundle, inputDataType, intermediateDataType);
        graphTensorBundle.dxTensor.memory().mark_device_modified();
        graphTensorBundle.dscaleTensor.memory().mark_device_modified();
        graphTensorBundle.dbiasTensor.memory().mark_device_modified();

        runCpuBatchnormBwd<InputType, IntermediateType>(cpuTensorBundle);

        Cpu_fp_reference_validation<InputType> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(cpuRefValidation.all_close(cpuTensorBundle.dxTensor.memory(),
                                               graphTensorBundle.dxTensor.memory()));

        Cpu_fp_reference_validation<IntermediateType> cpuRefIntermediateValidation(tolerance,
                                                                                   tolerance);
        EXPECT_TRUE(cpuRefIntermediateValidation.all_close(
            cpuTensorBundle.dscaleTensor.memory(), graphTensorBundle.dscaleTensor.memory()));
        EXPECT_TRUE(cpuRefIntermediateValidation.all_close(cpuTensorBundle.dbiasTensor.memory(),
                                                           graphTensorBundle.dbiasTensor.memory()));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

class BatchnormBackwardIntegrationTestBfloat16 : public BatchnormBackwardIntegrationTest
{
};

class BatchnormBackwardIntegrationTestHalf : public BatchnormBackwardIntegrationTest
{
};

class BatchnormBackwardIntegrationTestNHWC : public BatchnormBackwardIntegrationTest
{
};

namespace
{

std::vector<Batchnorm2dTestCase> getBnBwdTestCases()
{
    return {
        {.n = 1, .c = 3, .h = 14, .w = 14},
        {.n = 2, .c = 3, .h = 14, .w = 14},
        {.n = 64, .c = 3, .h = 14, .w = 14},
        {.n = 64, .c = 256, .h = 14, .w = 14},
        {.n = 64, .c = 256, .h = 28, .w = 28},
        {.n = 64, .c = 256, .h = 56, .w = 56},
        {.n = 64, .c = 512, .h = 14, .w = 14},
        {.n = 64, .c = 512, .h = 28, .w = 28},
        {.n = 64, .c = 512, .h = 7, .w = 7},
        {.n = 64, .c = 64, .h = 112, .w = 112},
        {.n = 64, .c = 64, .h = 56, .w = 56},
    };
}

} // namespace

TEST_P(BatchnormBackwardIntegrationTest, RunFloatBwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<float, float>(testCase, 4e-3f);
}

INSTANTIATE_TEST_SUITE_P(RunFloatBwdBatchnormGraph,
                         BatchnormBackwardIntegrationTest,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(BatchnormBackwardIntegrationTestBfloat16, RunBfloat16BwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<hip_bfloat16, float>(testCase, 4e-3_bf);
}

INSTANTIATE_TEST_SUITE_P(RunBfloat16BwdBatchnormGraph,
                         BatchnormBackwardIntegrationTestBfloat16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(BatchnormBackwardIntegrationTestHalf, RunHalfBwdBatchnormGraphNCWH)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<half, float>(testCase, 4e-3_h);
}

INSTANTIATE_TEST_SUITE_P(RunHalfBwdBatchnormGraph,
                         BatchnormBackwardIntegrationTestHalf,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(BatchnormBackwardIntegrationTestNHWC, RunFloatBwdBatchnormGraphNHWC)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBatchnormTest<float, float>(testCase, 4e-3f, Tensor_layout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(RunFloatBwdBatchnormGraphNHWC,
                         BatchnormBackwardIntegrationTestNHWC,
                         testing::ValuesIn(getBnBwdTestCases()));
