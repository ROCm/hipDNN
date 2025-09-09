// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/ReferenceImplementationInterface.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::reference_test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

class RefImplRegistryTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        registry = &getGlobalRegistry();
    }

    RefImplRegistry* registry;
};

TEST_F(RefImplRegistryTest, RegistryAutoPopulation)
{
    EXPECT_GT(registry->size(), 0);

    auto* convImpl = registry->get(
        NodeAttributes_ConvolutionFwdAttributes, DataType_FLOAT, DataType_FLOAT, DataType_FLOAT);
    EXPECT_NE(convImpl, nullptr);

    auto* bnImpl = registry->get(NodeAttributes_BatchnormInferenceAttributes,
                                 DataType_FLOAT,
                                 DataType_FLOAT,
                                 DataType_FLOAT);
    EXPECT_NE(bnImpl, nullptr);

    auto* bnBwdImpl = registry->get(NodeAttributes_BatchnormBackwardAttributes,
                                    DataType_DOUBLE,
                                    DataType_DOUBLE,
                                    DataType_DOUBLE);
    EXPECT_NE(bnBwdImpl, nullptr);
}

TEST_F(RefImplRegistryTest, InvalidOperationLookup)
{
    auto* invalidImpl = registry->get(
        static_cast<NodeAttributes>(999), DataType_FLOAT, DataType_FLOAT, DataType_FLOAT);
    EXPECT_EQ(invalidImpl, nullptr);

    auto* unsupportedImpl = registry->get(NodeAttributes_ConvolutionFwdAttributes,
                                          static_cast<DataType>(999),
                                          DataType_FLOAT,
                                          DataType_FLOAT);
    EXPECT_EQ(unsupportedImpl, nullptr);
}

TEST_F(RefImplRegistryTest, ConvolutionVariantPackExecution)
{
    Tensor<float> inputTensor({1, 1, 3, 3});
    Tensor<float> weightTensor({1, 1, 2, 2});
    Tensor<float> outputTensor({1, 1, 2, 2});

    inputTensor.setHostValue(0, 0, 0, 0, 1.0f);
    inputTensor.setHostValue(0, 0, 0, 1, 2.0f);
    inputTensor.setHostValue(0, 0, 0, 2, 3.0f);
    inputTensor.setHostValue(0, 0, 1, 0, 4.0f);
    inputTensor.setHostValue(0, 0, 1, 1, 5.0f);
    inputTensor.setHostValue(0, 0, 1, 2, 6.0f);
    inputTensor.setHostValue(0, 0, 2, 0, 7.0f);
    inputTensor.setHostValue(0, 0, 2, 1, 8.0f);
    inputTensor.setHostValue(0, 0, 2, 2, 9.0f);

    weightTensor.setHostValue(0, 0, 0, 0, 1.0f);
    weightTensor.setHostValue(0, 0, 0, 1, 0.0f);
    weightTensor.setHostValue(0, 0, 1, 0, 0.0f);
    weightTensor.setHostValue(0, 0, 1, 1, 1.0f);

    // Create flatbuffer node
    auto convNode = createConvolutionFwdNode(
        /*x_uid=*/1,
        /*w_uid=*/2,
        /*y_uid=*/3,
        /*strides=*/{1, 1},
        /*dilations=*/{1, 1},
        /*pre_padding=*/{0, 0},
        /*post_padding=*/{0, 0});

    // Create variant pack
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = &inputTensor; // x_tensor_uid
    variantPack[2] = &weightTensor; // w_tensor_uid
    variantPack[3] = &outputTensor; // y_tensor_uid

    // Execute via registry
    executeOperation(NodeAttributes_ConvolutionFwdAttributes,
                     DataType_FLOAT,
                     DataType_FLOAT,
                     DataType_FLOAT,
                     *convNode,
                     variantPack);

    // Verify execution completed (output should be modified)
    // With identity-like kernel [[1,0],[0,1]], expect output[0,0] = input[0,0] + input[1,1] = 1+5 = 6
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 6.0f, 1e-6f);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), 8.0f, 1e-6f); // 2+6
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), 12.0f, 1e-6f); // 4+8
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), 14.0f, 1e-6f); // 5+9
}

TEST_F(RefImplRegistryTest, BatchnormVariantPackExecution)
{
    Tensor<float> inputTensor({1, 1, 2, 2});
    Tensor<float> scaleTensor({1, 1, 1, 1});
    Tensor<float> biasTensor({1, 1, 1, 1});
    Tensor<float> meanTensor({1, 1, 1, 1});
    Tensor<float> varianceTensor({1, 1, 1, 1});
    Tensor<float> outputTensor({1, 1, 2, 2});

    inputTensor.setHostValue(0, 0, 0, 0, 1.0f);
    inputTensor.setHostValue(0, 0, 0, 1, 2.0f);
    inputTensor.setHostValue(0, 0, 1, 0, 3.0f);
    inputTensor.setHostValue(0, 0, 1, 1, 4.0f);

    scaleTensor.setHostValue(0, 0, 0, 0, 2.0f);
    biasTensor.setHostValue(0, 0, 0, 0, 0.5f);
    meanTensor.setHostValue(0, 0, 0, 0, 2.5f);
    varianceTensor.setHostValue(0, 0, 0, 0, 1.25f);

    // Create flatbuffer node
    auto bnNode = createBatchnormInferenceNode(
        /*x_uid=*/1,
        /*scale_uid=*/2,
        /*bias_uid=*/3,
        /*mean_uid=*/4,
        /*variance_uid=*/5,
        /*y_uid=*/6);

    // Create variant pack
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = &inputTensor; // x_tensor_uid
    variantPack[2] = &scaleTensor; // scale_tensor_uid
    variantPack[3] = &biasTensor; // bias_tensor_uid
    variantPack[4] = &meanTensor; // mean_tensor_uid
    variantPack[5] = &varianceTensor; // variance_tensor_uid
    variantPack[6] = &outputTensor; // y_tensor_uid

    // Execute via registry
    executeOperation(NodeAttributes_BatchnormInferenceAttributes,
                     DataType_FLOAT,
                     DataType_FLOAT,
                     DataType_FLOAT,
                     *bnNode,
                     variantPack);

    // Verify execution completed (output should be normalized)
    // Expected calculation: y = scale * (x - mean) / sqrt(var + eps) + bias
    // With mean=2.5, var=1.25, scale=2.0, bias=0.5, eps=1e-5
    // inv_variance = 1/sqrt(1.25 + 1e-5) ≈ 0.8944
    const std::vector<float> expectedOutput
        = {-2.18327084f, -0.39442361f, 1.39442361f, 3.18327084f};

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], 1e-5f);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput[1], 1e-5f);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput[2], 1e-5f);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput[3], 1e-5f);
}

TEST_F(RefImplRegistryTest, DoubleTypeExecution)
{
    Tensor<double> inputTensor({1, 1, 1, 1});
    Tensor<double> weightTensor({1, 1, 1, 1});
    Tensor<double> outputTensor({1, 1, 1, 1});

    inputTensor.setHostValue(0, 0, 0, 0, 3.0);
    weightTensor.setHostValue(0, 0, 0, 0, 4.0);

    auto convNode = createConvolutionFwdNode(
        /*x_uid=*/1,
        /*w_uid=*/2,
        /*y_uid=*/3,
        /*strides=*/{1, 1},
        /*dilations=*/{1, 1},
        /*pre_padding=*/{0, 0},
        /*post_padding=*/{0, 0});

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = &inputTensor;
    variantPack[2] = &weightTensor;
    variantPack[3] = &outputTensor;

    // Execute with double types
    executeOperation(NodeAttributes_ConvolutionFwdAttributes,
                     DataType_DOUBLE,
                     DataType_DOUBLE,
                     DataType_DOUBLE,
                     *convNode,
                     variantPack);

    // Verify 1x1 convolution: 3.0 * 4.0 = 12.0
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 12.0, 1e-10);
}

TEST_F(RefImplRegistryTest, InvalidVariantPackExecution)
{
    Tensor<float> inputTensor({1, 1, 1, 1});
    Tensor<float> weightTensor({1, 1, 1, 1});
    Tensor<float> outputTensor({1, 1, 1, 1});

    auto convNode = createConvolutionFwdNode(
        /*x_uid=*/1,
        /*w_uid=*/2,
        /*y_uid=*/3,
        /*strides=*/{1, 1},
        /*dilations=*/{1, 1},
        /*pre_padding=*/{0, 0},
        /*post_padding=*/{0, 0});

    // Missing weight tensor in variant pack
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = &inputTensor; // x_tensor_uid
    // variantPack[2] = &weightTensor; // MISSING!
    variantPack[3] = &outputTensor; // y_tensor_uid

    // Should throw when trying to access missing UID
    EXPECT_THROW(executeOperation(NodeAttributes_ConvolutionFwdAttributes,
                                  DataType_FLOAT,
                                  DataType_FLOAT,
                                  DataType_FLOAT,
                                  *convNode,
                                  variantPack),
                 std::out_of_range);
}

TEST_F(RefImplRegistryTest, UnsupportedOperationExecution)
{
    Tensor<float> inputTensor({1, 1, 1, 1});

    auto convNode = createConvolutionFwdNode(
        /*x_uid=*/1,
        /*w_uid=*/2,
        /*y_uid=*/3,
        /*strides=*/{1, 1},
        /*dilations=*/{1, 1},
        /*pre_padding=*/{0, 0},
        /*post_padding=*/{0, 0});

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = &inputTensor;

    // Try to execute with unsupported data type combination
    EXPECT_THROW(executeOperation(NodeAttributes_ConvolutionFwdAttributes,
                                  static_cast<DataType>(999),
                                  DataType_FLOAT,
                                  DataType_FLOAT,
                                  *convNode,
                                  variantPack),
                 std::runtime_error);
}

TEST_F(RefImplRegistryTest, BatchnormMissingOptionalTensors)
{
    Tensor<float> inputTensor({1, 1, 1, 1});
    Tensor<float> scaleTensor({1, 1, 1, 1});
    Tensor<float> biasTensor({1, 1, 1, 1});
    Tensor<float> outputTensor({1, 1, 1, 1});

    // Create batchnorm node with missing mean/variance UIDs
    auto bnNode = createBatchnormInferenceNode(
        /*x_uid=*/1,
        /*scale_uid=*/2,
        /*bias_uid=*/3,
        /*mean_uid=*/std::nullopt,
        /*variance_uid=*/std::nullopt,
        /*y_uid=*/6);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = &inputTensor;
    variantPack[2] = &scaleTensor;
    variantPack[3] = &biasTensor;
    variantPack[6] = &outputTensor;

    // Should throw error about missing required tensors
    EXPECT_THROW(executeOperation(NodeAttributes_BatchnormInferenceAttributes,
                                  DataType_FLOAT,
                                  DataType_FLOAT,
                                  DataType_FLOAT,
                                  *bnNode,
                                  variantPack),
                 std::runtime_error);
}

TEST_F(RefImplRegistryTest, RegistryPrintDiagnostics)
{
    EXPECT_NO_THROW(registry->printRegistered());
    EXPECT_GT(registry->size(), 0);
}

TEST_F(RefImplRegistryTest, IsApplicableChecks)
{
    auto* convImpl = registry->get(
        NodeAttributes_ConvolutionFwdAttributes, DataType_FLOAT, DataType_FLOAT, DataType_FLOAT);
    ASSERT_NE(convImpl, nullptr);

    // Create valid convolution node
    auto validConvNode = createConvolutionFwdNode(
        /*x_uid=*/1,
        /*w_uid=*/2,
        /*y_uid=*/3,
        /*strides=*/{1, 1},
        /*dilations=*/{1, 1},
        /*pre_padding=*/{0, 0},
        /*post_padding=*/{0, 0});

    EXPECT_TRUE(convImpl->isApplicable(*validConvNode));

    // Create batchnorm node (wrong type for conv implementation)
    auto bnNode = createBatchnormInferenceNode(
        /*x_uid=*/1,
        /*scale_uid=*/2,
        /*bias_uid=*/3,
        /*mean_uid=*/4,
        /*variance_uid=*/5,
        /*y_uid=*/6);

    EXPECT_FALSE(convImpl->isApplicable(*bnNode));
}

TEST_F(RefImplRegistryTest, TypeInfoReporting)
{
    auto* convImpl = registry->get(
        NodeAttributes_ConvolutionFwdAttributes, DataType_FLOAT, DataType_FLOAT, DataType_FLOAT);
    ASSERT_NE(convImpl, nullptr);

    std::string typeInfo = convImpl->getTypeInfo();
    EXPECT_FALSE(typeInfo.empty());
    EXPECT_NE(typeInfo.find("ConvolutionFwd"), std::string::npos);

    auto* bnImpl = registry->get(NodeAttributes_BatchnormInferenceAttributes,
                                 DataType_DOUBLE,
                                 DataType_DOUBLE,
                                 DataType_DOUBLE);
    ASSERT_NE(bnImpl, nullptr);

    typeInfo = bnImpl->getTypeInfo();
    EXPECT_FALSE(typeInfo.empty());
    EXPECT_NE(typeInfo.find("BatchnormInference"), std::string::npos);
}

TEST_F(RefImplRegistryTest, DeviceAwareLookup)
{
    auto* cpuImpl = registry->get(NodeAttributes_ConvolutionFwdAttributes,
                                  DataType_FLOAT,
                                  DataType_FLOAT,
                                  DataType_FLOAT,
                                  DeviceType::CPU);
    EXPECT_NE(cpuImpl, nullptr);

    auto* gpuImpl = registry->get(NodeAttributes_ConvolutionFwdAttributes,
                                  DataType_FLOAT,
                                  DataType_FLOAT,
                                  DataType_FLOAT,
                                  DeviceType::GPU);
    EXPECT_EQ(gpuImpl, nullptr);
}

TEST_F(RefImplRegistryTest, BackwardCompatibilityDefaultsToCpu)
{
    auto* implOldApi = registry->get(
        NodeAttributes_ConvolutionFwdAttributes, DataType_FLOAT, DataType_FLOAT, DataType_FLOAT);

    auto* implNewApiCpu = registry->get(NodeAttributes_ConvolutionFwdAttributes,
                                        DataType_FLOAT,
                                        DataType_FLOAT,
                                        DataType_FLOAT,
                                        DeviceType::CPU);

    EXPECT_EQ(implOldApi, implNewApiCpu);
    EXPECT_NE(implOldApi, nullptr);
}

TEST_F(RefImplRegistryTest, DeviceAwarePrintDiagnostics)
{
    std::ostringstream captured;
    std::streambuf* orig = std::cout.rdbuf(captured.rdbuf());

    registry->printRegistered();

    std::cout.rdbuf(orig);
    std::string output = captured.str();

    EXPECT_NE(output.find("[CPU]"), std::string::npos);
    EXPECT_GT(output.length(), 0);
}

TEST_F(RefImplRegistryTest, ManualDeviceRegistration)
{
    RefImplRegistry testRegistry;

    testRegistry.registerImpl<NodeAttributes_ConvolutionFwdAttributes,
                              DataType_FLOAT,
                              DataType_FLOAT,
                              DataType_FLOAT,
                              CpuImplTraits>(DeviceType::CPU);

    auto* cpuImpl = testRegistry.get(NodeAttributes_ConvolutionFwdAttributes,
                                     DataType_FLOAT,
                                     DataType_FLOAT,
                                     DataType_FLOAT,
                                     DeviceType::CPU);
    EXPECT_NE(cpuImpl, nullptr);

    auto* gpuImpl = testRegistry.get(NodeAttributes_ConvolutionFwdAttributes,
                                     DataType_FLOAT,
                                     DataType_FLOAT,
                                     DataType_FLOAT,
                                     DeviceType::GPU);
    EXPECT_EQ(gpuImpl, nullptr);

    EXPECT_EQ(testRegistry.size(), 1);
}

TEST_F(RefImplRegistryTest, DeviceAwareExecuteOperation)
{
    Tensor<float> inputTensor({1, 1, 1, 1});
    Tensor<float> weightTensor({1, 1, 1, 1});
    Tensor<float> outputTensor({1, 1, 1, 1});

    inputTensor.setHostValue(0, 0, 0, 0, 5.0f);
    weightTensor.setHostValue(0, 0, 0, 0, 7.0f);

    auto convNode = createConvolutionFwdNode(
        /*x_uid=*/1,
        /*w_uid=*/2,
        /*y_uid=*/3,
        /*strides=*/{1, 1},
        /*dilations=*/{1, 1},
        /*pre_padding=*/{0, 0},
        /*post_padding=*/{0, 0});

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = &inputTensor;
    variantPack[2] = &weightTensor;
    variantPack[3] = &outputTensor;

    EXPECT_NO_THROW(executeOperation(NodeAttributes_ConvolutionFwdAttributes,
                                     DataType_FLOAT,
                                     DataType_FLOAT,
                                     DataType_FLOAT,
                                     *convNode,
                                     variantPack));

    // Verify result: 5.0 * 7.0 = 35.0
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 35.0f, 1e-6f);
}
