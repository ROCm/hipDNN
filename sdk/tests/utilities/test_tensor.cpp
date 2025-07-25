// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/utilities/tensor.hpp>

using namespace hipdnn_sdk::utilities;

TEST(TestTensor, BasicRowMajorUsage)
{
    Tensor tensor = Tensor::make_test_tensor<float>({1, 2, 3, 4});

    EXPECT_EQ(tensor.memory().count(), 24);
    EXPECT_EQ(tensor.strides()[0], 24);
    EXPECT_EQ(tensor.strides()[1], 12);
    EXPECT_EQ(tensor.strides()[2], 4);
    EXPECT_EQ(tensor.strides()[3], 1);
}

TEST(TestTensor, BasicColumnMajorUsage)
{
    Tensor tensor = Tensor::make_test_tensor<float>({1, 2, 3, 4}, false);

    EXPECT_EQ(tensor.memory().count(), 24);
    EXPECT_EQ(tensor.strides()[0], 1);
    EXPECT_EQ(tensor.strides()[1], 1);
    EXPECT_EQ(tensor.strides()[2], 2);
    EXPECT_EQ(tensor.strides()[3], 6);
}

TEST(TestTensor, FillWithValuesUsage)
{
    Tensor tensor = Tensor::make_test_tensor<float>({1, 2, 3, 4}, false);

    tensor.fill_with_value(1.0f);
    float* buffer = tensor.memory().host_data<float>();

    for(size_t i = 0; i < tensor.memory().count(); i++)
    {
        EXPECT_FLOAT_EQ(1.0f, buffer[i]);
    }
}

TEST(TestTensor, FillWithRandomValuesUsage)
{
    Tensor tensor = Tensor::make_test_tensor<float>({1, 2, 3, 4}, false);

    tensor.fill_with_random_values(1.0f, 3.0f);
    float* buffer = tensor.memory().host_data<float>();

    for(size_t i = 0; i < tensor.memory().count(); i++)
    {
        EXPECT_GE(buffer[i], 1.0f);
        EXPECT_LE(buffer[i], 3.0f);
    }
}