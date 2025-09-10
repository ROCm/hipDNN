// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/ReferenceImplementationInterface.hpp>

#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

TEST(TestCpuReferenceContainer, BatchnormInferFloatUsage)
{
    Tensor<float> inputTensor({1, 3, 224, 224});
    Tensor<float> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuReferenceContainer, BatchnormInferBFloat16Usage)
{
    Tensor<hip_bfloat16> inputTensor({1, 3, 224, 224});
    Tensor<hip_bfloat16> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuReferenceContainer, BatchnormInferHalfUsage)
{
    Tensor<half> inputTensor({1, 3, 224, 224});
    Tensor<half> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuReferenceContainer, BatchnormInferDoubleUsage)
{
    Tensor<double> inputTensor({1, 3, 224, 224});
    Tensor<double> outputTensor({1, 3, 224, 224});
    Tensor<double> biasTensor({1, 3});
    Tensor<double> scaleTensor({1, 3});
    Tensor<double> meanTensor({1, 3});
    Tensor<double> varianceTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuReferenceContainer, BatchnormInferFloatUsageNhwc)
{
    Tensor<float> inputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> outputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuReferenceContainer, BatchnormInferSanityValidation)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    // x = [1, 2, 3, 4]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    // fixed scale and bias parameters (one channel)
    scaleTensor.setHostValue(2.0, 0, 0);
    biasTensor.setHostValue(0.5, 0, 0);

    // inference uses population statistics per channel:
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 5.0 / 4 = 1.25
    // (in practice, computed during training)
    meanTensor.setHostValue(2.5, 0, 0);
    varianceTensor.setHostValue(1.25, 0, 0);

    // output is calculated via a pointwise linear transform on x:
    // y = scale * (x - mean) * inv_variance + bias = 2 * (x - 2.5) * inv_variance + 0.5
    // where inv_variance (named by convention) = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    const std::vector<double> expectedOutput = {-2.18327084, -0.39442361, 1.39442361, 3.18327084};

    CpuReferenceContainer refImpl;
    refImpl.batchnormFwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput[1], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput[2], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput[3], tolerance);
}

TEST(TestCpuReferenceContainer, BatchnormBwdFloatUsage)
{
    Tensor<float> xTensor({6, 3, 32, 32});
    Tensor<float> dyTensor({6, 3, 32, 32});
    Tensor<float> dxTensor({6, 3, 32, 32});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(TestCpuReferenceContainer, BatchnormBwdBFloat16Usage)
{
    Tensor<hip_bfloat16> xTensor({6, 3, 32, 32});
    Tensor<hip_bfloat16> dyTensor({6, 3, 32, 32});
    Tensor<hip_bfloat16> dxTensor({6, 3, 32, 32});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(TestCpuReferenceContainer, BatchnormBwdHalfUsage)
{
    Tensor<half> xTensor({6, 3, 32, 32});
    Tensor<half> dyTensor({6, 3, 32, 32});
    Tensor<half> dxTensor({6, 3, 32, 32});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(TestCpuReferenceContainer, BatchnormBwdDoubleUsage)
{
    Tensor<double> xTensor({6, 3, 32, 32});
    Tensor<double> dyTensor({6, 3, 32, 32});
    Tensor<double> dxTensor({6, 3, 32, 32});
    Tensor<double> scaleTensor({1, 3});
    Tensor<double> meanTensor({1, 3});
    Tensor<double> invVarianceTensor({1, 3});
    Tensor<double> dscaleTensor({1, 3});
    Tensor<double> dbiasTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(TestCpuReferenceContainer, BatchnormBwdFloatUsageNhwc)
{
    Tensor<float> xTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> dyTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> dxTensor({6, 3, 32, 32});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});
    Tensor<float> dscaleTensor({1, 3});
    Tensor<float> dbiasTensor({1, 3});

    CpuReferenceContainer refImpl;

    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);
}

TEST(TestCpuReferenceContainer, BatchnormBwdSanityValidation)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> xTensor(dims);
    Tensor<double> dyTensor(dims);
    Tensor<double> dxTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> invVarianceTensor({1, 1});
    Tensor<double> dscaleTensor({1, 1});
    Tensor<double> dbiasTensor({1, 1});

    // x = [1, 2, 3, 4]
    xTensor.setHostValue(1.0, 0, 0, 0, 0);
    xTensor.setHostValue(2.0, 0, 0, 0, 1);
    xTensor.setHostValue(3.0, 0, 0, 1, 0);
    xTensor.setHostValue(4.0, 0, 0, 1, 1);

    // gradient dy = [0.1, 0.2, 0.3, 0.4]
    dyTensor.setHostValue(0.1, 0, 0, 0, 0);
    dyTensor.setHostValue(0.2, 0, 0, 0, 1);
    dyTensor.setHostValue(0.3, 0, 0, 1, 0);
    dyTensor.setHostValue(0.4, 0, 0, 1, 1);

    // scale (one channel) = 2.0
    scaleTensor.setHostValue(2.0, 0, 0);

    // 1 batch, so compute mean and variance over all elements
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 5.0 / 4 = 1.25
    // inv_variance = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    meanTensor.setHostValue(2.5, 0, 0);
    invVarianceTensor.setHostValue(0.894423613312618, 0, 0);

    // dbias = sum(dy) = 0.1 + 0.2 + 0.3 + 0.4 = 1.0
    auto expectedDbias = 1.0;

    // x_hat = (x - mean) * inv_variance = [-1.34163542 -0.44721181  0.44721181  1.34163542]
    // dscale = sum(dy * x_hat) = sum([-1.34163542 -0.44721181  0.44721181  1.34163542]) = 0.447211806656309
    auto expectedDscale = 0.447211806656309;

    // dx is calculated pointwise via the full backward formula
    // dx = scale * inv_variance * (dy - mean(dy) - x_hat * dscale / 4)
    std::vector<double> expectedDx
        = {-2.14659950e-06, -7.15533166e-07, 7.15533166e-07, 2.14659950e-06};

    CpuReferenceContainer refImpl;
    refImpl.batchnormBwd(dyTensor,
                         xTensor,
                         meanTensor,
                         invVarianceTensor,
                         scaleTensor,
                         dxTensor,
                         dscaleTensor,
                         dbiasTensor);

    auto tolerance = 1e-6;

    EXPECT_NEAR(dbiasTensor.getHostValue(0, 0), expectedDbias, tolerance);
    EXPECT_NEAR(dscaleTensor.getHostValue(0, 0), expectedDscale, tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 0), expectedDx[0], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 0, 1), expectedDx[1], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 0), expectedDx[2], tolerance);
    EXPECT_NEAR(dxTensor.getHostValue(0, 0, 1, 1), expectedDx[3], tolerance);
}

// Convolution Forward Inference Tests

TEST(TestCpuReferenceContainer, ConvFwdFloatUsage)
{
    // Basic 2D convolution: 1 batch, 2 input channels, 3 output channels, 1 group
    // Input: 1x2x4x4, Weight: 3x2x3x3, Output: 1x3x2x2
    Tensor<float> inputTensor({1, 2, 4, 4}); // NCHW
    Tensor<float> weightTensor({3, 2, 3, 3}); // [G*K][C][Y][X] - 4D flattened
    Tensor<float> outputTensor({1, 3, 2, 2}); // NCHW

    std::vector<int64_t> strides = {1, 1}; // [H, W]
    std::vector<int64_t> dilations = {1, 1}; // [H, W]
    std::vector<int64_t> padding = {0, 0}; // [H, W]

    CpuReferenceContainer refImpl;

    refImpl.convFwdInference(inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuReferenceContainer, ConvFwdDoubleUsage)
{
    Tensor<double> inputTensor({2, 4, 8, 8});
    Tensor<double> weightTensor({8, 4, 3, 3}); // 4D: [G*K][C][Y][X] = [8][4][3][3]
    Tensor<double> outputTensor({2, 8, 6, 6});

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    refImpl.convFwdInference(inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuReferenceContainer, ConvFwdHalfUsage)
{
    Tensor<half> inputTensor({1, 1, 5, 5});
    Tensor<half> weightTensor({1, 1, 3, 3}); // 4D: [G*K][C][Y][X] = [1][1][3][3]
    Tensor<half> outputTensor({1, 1, 3, 3});

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    refImpl.convFwdInference(inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuReferenceContainer, ConvFwdBFloat16Usage)
{
    Tensor<hip_bfloat16> inputTensor({1, 3, 32, 32});
    Tensor<hip_bfloat16> weightTensor({16, 3, 5, 5}); // 4D: [G*K][C][Y][X] = [16][3][5][5]
    Tensor<hip_bfloat16> outputTensor({1, 16, 28, 28});

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    refImpl.convFwdInference(inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuReferenceContainer, ConvFwdWithStridesAndPadding)
{
    // Test with strides=2, padding=1
    Tensor<float> inputTensor({1, 1, 4, 4});
    Tensor<float> weightTensor({1, 1, 3, 3}); // 4D: [G*K][C][Y][X] = [1][1][3][3]
    Tensor<float> outputTensor({1, 1, 2, 2}); // (4 + 2*1 - 3)/2 + 1 = 2

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {1, 1};

    CpuReferenceContainer refImpl;

    refImpl.convFwdInference(inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuReferenceContainer, ConvFwdWithDilations)
{
    // Test with dilations=2
    Tensor<float> inputTensor({1, 1, 5, 5});
    Tensor<float> weightTensor({1, 1, 3, 3}); // 4D: [G*K][C][Y][X] = [1][1][3][3]
    Tensor<float> outputTensor({1, 1, 1, 1}); // (5 - (3-1)*2 - 1)/1 + 1 = 1

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {2, 2};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    refImpl.convFwdInference(inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuReferenceContainer, ConvFwdSanityValidation)
{
    // Simple 1x1 input, 1x1 kernel test for verification
    Tensor<double> inputTensor({1, 1, 1, 1});
    Tensor<double> weightTensor({1, 1, 1, 1}); // 4D: [G*K][C][Y][X] = [1][1][1][1]
    Tensor<double> outputTensor({1, 1, 1, 1});

    // Set input value to 2.0
    inputTensor.setHostValue(2.0, 0, 0, 0, 0);

    // Set weight value to 3.0 (linearized indexing: G*K=0, C=0, Y=0, X=0)
    weightTensor.setHostValue(3.0, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    refImpl.convFwdInference(inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Expected output: 2.0 * 3.0 = 6.0
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 6.0, 1e-10);
}

TEST(TestCpuReferenceContainer, ConvFwdDetailedValidation)
{
    // 2x2 input with 2x2 kernel
    Tensor<double> inputTensor({1, 1, 2, 2});
    Tensor<double> weightTensor({1, 1, 2, 2}); // 4D: [G*K][C][Y][X] = [1][1][2][2]
    Tensor<double> outputTensor({1, 1, 1, 1});

    // Input: [[1, 2], [3, 4]]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    // Weight: [[1, 0], [0, 1]] (identity-like kernel)
    // 4D indexing: [G*K=0][C=0][Y][X]
    weightTensor.setHostValue(1.0, 0, 0, 0, 0); // Y=0, X=0
    weightTensor.setHostValue(0.0, 0, 0, 0, 1); // Y=0, X=1
    weightTensor.setHostValue(0.0, 0, 0, 1, 0); // Y=1, X=0
    weightTensor.setHostValue(1.0, 0, 0, 1, 1); // Y=1, X=1

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    refImpl.convFwdInference(inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Expected: 1*1 + 2*0 + 3*0 + 4*1 = 5.0
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 5.0, 1e-10);
}

// Parameter validation tests
TEST(TestCpuReferenceContainer, ConvFwdInvalidInputDimensions)
{
    Tensor<float> inputTensor({1, 2, 4}); // 3D instead of 4D
    Tensor<float> weightTensor({1, 1, 2, 3, 3});
    Tensor<float> outputTensor({1, 1, 2, 2});

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    EXPECT_THROW(refImpl.convFwdInference(
                     inputTensor, weightTensor, outputTensor, strides, dilations, padding),
                 std::invalid_argument);
}

TEST(TestCpuReferenceContainer, ConvFwdInvalidWeightDimensions)
{
    Tensor<float> inputTensor({1, 2, 4, 4});
    Tensor<float> weightTensor({1, 1, 2}); // 3D instead of 4D
    Tensor<float> outputTensor({1, 1, 2, 2});

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    EXPECT_THROW(refImpl.convFwdInference(
                     inputTensor, weightTensor, outputTensor, strides, dilations, padding),
                 std::invalid_argument);
}

TEST(TestCpuReferenceContainer, ConvFwdInvalidStrideSize)
{
    Tensor<float> inputTensor({1, 2, 4, 4});
    Tensor<float> weightTensor({1, 2, 3, 3}); // 4D: [G*K][C][Y][X] = [1][2][3][3]
    Tensor<float> outputTensor({1, 1, 2, 2});

    std::vector<int64_t> strides = {1}; // Should be size 2
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    EXPECT_THROW(refImpl.convFwdInference(
                     inputTensor, weightTensor, outputTensor, strides, dilations, padding),
                 std::invalid_argument);
}

TEST(TestCpuReferenceContainer, ConvFwdInvalidStrideValue)
{
    Tensor<float> inputTensor({1, 2, 4, 4});
    Tensor<float> weightTensor({1, 2, 3, 3}); // 4D: [G*K][C][Y][X] = [1][2][3][3]
    Tensor<float> outputTensor({1, 1, 2, 2});

    std::vector<int64_t> strides = {0, 1}; // Zero stride is invalid
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    EXPECT_THROW(refImpl.convFwdInference(
                     inputTensor, weightTensor, outputTensor, strides, dilations, padding),
                 std::invalid_argument);
}

TEST(TestCpuReferenceContainer, ConvFwdInvalidDilationValue)
{
    Tensor<float> inputTensor({1, 2, 4, 4});
    Tensor<float> weightTensor({1, 2, 3, 3}); // 4D: [G*K][C][Y][X] = [1][2][3][3]
    Tensor<float> outputTensor({1, 1, 2, 2});

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {-1, 1}; // Negative dilation is invalid
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    EXPECT_THROW(refImpl.convFwdInference(
                     inputTensor, weightTensor, outputTensor, strides, dilations, padding),
                 std::invalid_argument);
}

TEST(TestCpuReferenceContainer, ConvFwdInvalidPaddingValue)
{
    Tensor<float> inputTensor({1, 2, 4, 4});
    Tensor<float> weightTensor({1, 2, 3, 3}); // 4D: [G*K][C][Y][X] = [1][2][3][3]
    Tensor<float> outputTensor({1, 1, 2, 2});

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {-1, 0}; // Negative padding is invalid

    CpuReferenceContainer refImpl;

    EXPECT_THROW(refImpl.convFwdInference(
                     inputTensor, weightTensor, outputTensor, strides, dilations, padding),
                 std::invalid_argument);
}

// NHWC Layout Tests

TEST(TestCpuReferenceContainer, ConvFwdFloatUsageNhwc)
{
    // Basic 2D convolution with NHWC layout
    Tensor<float> inputTensor({1, 2, 4, 4}, TensorLayout::NHWC);
    Tensor<float> weightTensor({3, 2, 3, 3}); // Weight layout remains [G*K][C][Y][X]
    Tensor<float> outputTensor({1, 3, 2, 2}, TensorLayout::NHWC);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    CpuReferenceContainer refImpl;

    refImpl.convFwdInference(inputTensor, weightTensor, outputTensor, strides, dilations, padding);
}

TEST(TestCpuReferenceContainer, ConvFwdNhwcLayoutValidation)
{
    // Test that NCHW and NHWC produce equivalent results
    // Simple 1x1 input, 1x1 kernel for easy validation

    // NCHW tensors
    Tensor<double> inputTensorNCHW({1, 1, 1, 1}, TensorLayout::NCHW);
    Tensor<double> outputTensorNCHW({1, 1, 1, 1}, TensorLayout::NCHW);

    // NHWC tensors
    Tensor<double> inputTensorNHWC({1, 1, 1, 1}, TensorLayout::NHWC);
    Tensor<double> outputTensorNHWC({1, 1, 1, 1}, TensorLayout::NHWC);

    // Shared weight tensor (layout doesn't change)
    Tensor<double> weightTensor({1, 1, 1, 1}); // [G*K][C][Y][X] = [1][1][1][1]

    // Set identical input values
    inputTensorNCHW.setHostValue(2.5, 0, 0, 0, 0);
    inputTensorNHWC.setHostValue(2.5, 0, 0, 0, 0);

    // Set weight value
    weightTensor.setHostValue(3.5, 0, 0, 0, 0);

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    // NEW ARCHITECTURE: Layout equivalence validation!
    CpuReferenceContainer refImpl;

    // Run convolution on both layouts
    refImpl.convFwdInference(
        inputTensorNCHW, weightTensor, outputTensorNCHW, strides, dilations, padding);
    refImpl.convFwdInference(
        inputTensorNHWC, weightTensor, outputTensorNHWC, strides, dilations, padding);

    // Results should be identical
    double expectedResult = 2.5 * 3.5; // 8.75
    EXPECT_NEAR(outputTensorNCHW.getHostValue(0, 0, 0, 0), expectedResult, 1e-10);
    EXPECT_NEAR(outputTensorNHWC.getHostValue(0, 0, 0, 0), expectedResult, 1e-10);

    // Verify both layouts produce the same result
    EXPECT_NEAR(outputTensorNCHW.getHostValue(0, 0, 0, 0),
                outputTensorNHWC.getHostValue(0, 0, 0, 0),
                1e-10);
}
