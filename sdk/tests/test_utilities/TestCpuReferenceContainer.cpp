// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/ReferenceImplementationInterface.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::reference_test_utilities;
using namespace hipdnn_sdk::utilities;

class CpuReferenceContainerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup test data
    }
};

TEST_F(CpuReferenceContainerTest, BatchnormFwdInferenceUsage)
{
    // Create tensors
    Tensor<float> inputTensor({1, 3, 224, 224});
    Tensor<float> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    // NEW ARCHITECTURE: Use CpuReferenceContainer
    CpuReferenceContainer container;

    // Template parameters are on each function call, not on the class
    container.batchnormFwdInference<float, float, float>(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);

    // Test passes if no exception is thrown
    SUCCEED();
}

TEST_F(CpuReferenceContainerTest, ConvolutionFwdInferenceUsage)
{
    // Basic 2D convolution: 1 batch, 2 input channels, 3 output channels
    Tensor<float> inputTensor({1, 2, 4, 4});
    Tensor<float> weightTensor({3, 2, 3, 3}); // [G*K][C][Y][X]
    Tensor<float> outputTensor({1, 3, 2, 2});

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    // NEW ARCHITECTURE: Use CpuReferenceContainer
    CpuReferenceContainer container;

    // Template parameters are on each function call, not on the class
    container.convFwdInference<float>(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    // Test passes if no exception is thrown
    SUCCEED();
}

TEST_F(CpuReferenceContainerTest, MixedDataTypesUsage)
{
    // Demonstrate mixed precision: half input, float scale/bias
    Tensor<half> inputTensor({1, 3, 32, 32});
    Tensor<half> outputTensor({1, 3, 32, 32});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    CpuReferenceContainer container;

    // Template allows different data type combinations
    container.batchnormFwdInference<half, float, float>(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);

    SUCCEED();
}

TEST_F(CpuReferenceContainerTest, TemplateParametersOnMethods)
{
    // Demonstrate template parameters are on individual methods, not the class
    CpuReferenceContainer container;

    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 2, 2});
    Tensor<float> biasTensor({1, 1});
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> meanTensor({1, 1});
    Tensor<float> varianceTensor({1, 1});

    // Templates are on each method call - exactly what was requested in PR feedback
    container.batchnormFwdInference<float, float, float>(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);

    SUCCEED();
}

TEST_F(CpuReferenceContainerTest, DirectImplementationUsage)
{
    // Demonstrate the simplified architecture using implementation classes directly

    // CpuReferenceContainer is now just a type alias:
    // using CpuReferenceContainer = BaseReferenceContainer<CpuFpReferenceBatchnormImpl, CpuFpReferenceConvolutionImpl>;
    CpuReferenceContainer container;

    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 2, 2});
    Tensor<float> biasTensor({1, 1});
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> meanTensor({1, 1});
    Tensor<float> varianceTensor({1, 1});

    // SIMPLIFIED: This uses implementation classes directly, no wrapper layers!
    container.batchnormFwdInference<float, float, float>(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);

    SUCCEED();
}

TEST_F(CpuReferenceContainerTest, SimplifiedArchitecture)
{
    // Show how the simplified architecture works:

    // One template implementation:
    // template<template<class...> class BatchnormRefType, template<class...> class ConvRefType>
    // class BaseReferenceContainer { ... }

    // Adding new backends is just ONE LINE with implementation classes:
    // using CpuReferenceContainer = BaseReferenceContainer<CpuFpReferenceBatchnormImpl, CpuFpReferenceConvolutionImpl>;
    // using GpuReferenceContainer = BaseReferenceContainer<GpuFpReferenceBatchnormImpl, GpuFpReferenceConvolutionImpl>;

    CpuReferenceContainer container;

    Tensor<float> inputTensor({1, 3, 4, 4});
    Tensor<float> weightTensor({2, 3, 3, 3});
    Tensor<float> outputTensor({1, 2, 2, 2});

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> padding = {0, 0};

    // Clean API calling implementation classes directly!
    container.convFwdInference<float>(
        inputTensor, weightTensor, outputTensor, strides, dilations, padding);

    SUCCEED();
}

TEST_F(CpuReferenceContainerTest, FutureExtensibility)
{
    // Show how easy it will be to add GPU support
    CpuReferenceContainer container;

    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 2, 2});
    Tensor<float> biasTensor({1, 1});
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> meanTensor({1, 1});
    Tensor<float> varianceTensor({1, 1});

    // When GPU support is added, it will be:
    // using GpuReferenceContainer = BaseReferenceContainer<GpuFpReferenceBatchnormImpl, GpuFpReferenceConvolutionImpl>;
    // Same API, different backend!
    container.batchnormFwdInference<float, float, float>(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);

    SUCCEED();
}
