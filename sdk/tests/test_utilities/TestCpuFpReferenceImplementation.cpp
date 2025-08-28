// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceImplementation.hpp>

#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/HalfUtils.hpp>
#include <hipdnn_sdk/utilities/HipBfloat16Utils.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::reference_test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

TEST(CpuFpReferenceImplementation, BatchnormInferFloatUsage)
{
    Tensor<float> inputTensor({1, 3, 224, 224});
    Tensor<float> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuFpReferenceImplementation<float, float, float> refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BatchnormInferBFloat16Usage)
{
    Tensor<hip_bfloat16> inputTensor({1, 3, 224, 224});
    Tensor<hip_bfloat16> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuFpReferenceImplementation<hip_bfloat16, float, float> refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BatchnormInferHalfUsage)
{
    Tensor<half> inputTensor({1, 3, 224, 224});
    Tensor<half> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuFpReferenceImplementation<half, float, float> refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BatchnormInferDoubleUsage)
{
    Tensor<double> inputTensor({1, 3, 224, 224});
    Tensor<double> outputTensor({1, 3, 224, 224});
    Tensor<double> biasTensor({1, 3});
    Tensor<double> scaleTensor({1, 3});
    Tensor<double> meanTensor({1, 3});
    Tensor<double> varianceTensor({1, 3});

    CpuFpReferenceImplementation<double, double, double> refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BatchnormInferFloatUsageNHWC)
{
    Tensor<float> inputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> outputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> biasTensor({1, 3, 1, 1});
    Tensor<float> scaleTensor({1, 3, 1, 1});
    Tensor<float> meanTensor({1, 3, 1, 1});
    Tensor<float> varianceTensor({1, 3, 1, 1});

    CpuFpReferenceImplementation<float, float, float> refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(CpuFpReferenceImplementation, BatchnormInferSanityValidation)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1, 1, 1});
    Tensor<double> biasTensor({1, 1, 1, 1});
    Tensor<double> meanTensor({1, 1, 1, 1});
    Tensor<double> varianceTensor({1, 1, 1, 1});

    // x = [1, 2, 3, 4]
    inputTensor.setHostValue(0, 0, 0, 0, 1.0);
    inputTensor.setHostValue(0, 0, 0, 1, 2.0);
    inputTensor.setHostValue(0, 0, 1, 0, 3.0);
    inputTensor.setHostValue(0, 0, 1, 1, 4.0);

    // fixed scale and bias parameters (one channel)
    scaleTensor.setHostValue(0, 0, 0, 0, 2.0);
    biasTensor.setHostValue(0, 0, 0, 0, 0.5);

    // inference uses population statistics per channel:
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 5.0 / 4 = 1.25
    // (in practice, computed during training)
    meanTensor.setHostValue(0, 0, 0, 0, 2.5);
    varianceTensor.setHostValue(0, 0, 0, 0, 1.25);

    // output is calculated via a pointwise linear transform on x:
    // y = scale * (x - mean) * inv_variance + bias = 2 * (x - 2.5) * inv_variance + 0.5
    // where inv_variance (named by convention) = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    const std::vector<double> expectedOutput = {-2.18327084, -0.39442361, 1.39442361, 3.18327084};

    CpuFpReferenceImplementation<double, double, double> refImpl;
    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput[1], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput[2], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput[3], tolerance);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdFloatUsage)
{
    Tensor<float> xTensor({6, 3, 32, 32});
    Tensor<float> dyTensor({6, 3, 32, 32});
    Tensor<float> dxTensor({6, 3, 32, 32});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    CpuFpReferenceImplementation<float, float, float> refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdBFloat16Usage)
{
    Tensor<hip_bfloat16> xTensor({6, 3, 32, 32});
    Tensor<hip_bfloat16> dyTensor({6, 3, 32, 32});
    Tensor<hip_bfloat16> dxTensor({6, 3, 32, 32});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    CpuFpReferenceImplementation<hip_bfloat16, float, float> refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdHalfUsage)
{
    Tensor<half> xTensor({6, 3, 32, 32});
    Tensor<half> dyTensor({6, 3, 32, 32});
    Tensor<half> dxTensor({6, 3, 32, 32});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    CpuFpReferenceImplementation<half, float, float> refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdDoubleUsage)
{
    Tensor<double> xTensor({6, 3, 32, 32});
    Tensor<double> dyTensor({6, 3, 32, 32});
    Tensor<double> dxTensor({6, 3, 32, 32});
    Tensor<double> scaleTensor({1, 3});
    Tensor<double> meanTensor({1, 3});
    Tensor<double> invVarianceTensor({1, 3});
    Tensor<double> dscaleTensor({1, 3});
    Tensor<double> dbiasTensor({1, 3});

    CpuFpReferenceImplementation<double, double, double> refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdFloatUsageNHWC)
{
    Tensor<float> xTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> dyTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> dxTensor({6, 3, 32, 32});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    CpuFpReferenceImplementation<float, float, float> refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(CpuFpReferenceImplementation, BatchnormBwdSanityValidation)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> xTensor(dims);
    Tensor<double> dyTensor(dims);
    Tensor<double> dxTensor(dims);
    Tensor<double> scaleTensor({1, 1, 1, 1});
    Tensor<double> meanTensor({1, 1, 1, 1});
    Tensor<double> invVarianceTensor({1, 1, 1, 1});
    Tensor<double> dscaleTensor({1, 1, 1, 1});
    Tensor<double> dbiasTensor({1, 1, 1, 1});

    // x = [1, 2, 3, 4]
    xTensor.setHostValue(0, 0, 0, 0, 1.0);
    xTensor.setHostValue(0, 0, 0, 1, 2.0);
    xTensor.setHostValue(0, 0, 1, 0, 3.0);
    xTensor.setHostValue(0, 0, 1, 1, 4.0);

    // gradient dy = [0.1, 0.2, 0.3, 0.4]
    dyTensor.setHostValue(0, 0, 0, 0, 0.1);
    dyTensor.setHostValue(0, 0, 0, 1, 0.2);
    dyTensor.setHostValue(0, 0, 1, 0, 0.3);
    dyTensor.setHostValue(0, 0, 1, 1, 0.4);

    // scale (one channel) = 2.0
    scaleTensor.setHostValue(0, 0, 0, 0, 2.0);

    // 1 batch, so compute mean and variance over all elements
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 5.0 / 4 = 1.25
    // inv_variance = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    meanTensor.setHostValue(0, 0, 0, 0, 2.5);
    invVarianceTensor.setHostValue(0, 0, 0, 0, 0.894423613312618);

    // dbias = sum(dy) = 0.1 + 0.2 + 0.3 + 0.4 = 1.0
    auto expectedDbias = 1.0;

    // x_hat = (x - mean) * inv_variance = [-1.34163542 -0.44721181  0.44721181  1.34163542]
    // dscale = sum(dy * x_hat) = sum([-1.34163542 -0.44721181  0.44721181  1.34163542]) = 0.447211806656309
    auto expectedDscale = 0.447211806656309;

    // dx is calculated pointwise via the full backward formula
    // dx = scale * inv_variance * (dy - mean(dy) - x_hat * dscale / 4)
    std::vector<double> expectedDx
        = {-2.14659950e-06, -7.15533166e-07, 7.15533166e-07, 2.14659950e-06};

    CpuFpReferenceImplementation<double, double, double> refImpl;
    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);

    auto tolerance = 1e-6;

    EXPECT_NEAR(dbiasTensor.getHostValue(0, 0, 0, 0), expectedDbias, tolerance);
    EXPECT_NEAR(dscaleTensor.getHostValue(0, 0, 0, 0), expectedDscale, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 0), expectedDx[0], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 1), expectedDx[1], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 0), expectedDx[2], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 1), expectedDx[3], tolerance);
}
