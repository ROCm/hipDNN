// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::utilities;

TEST(TestTensor, BasicRowMajorUsage)
{
    Tensor<float> tensor({1, 2, 3, 4});

    // NCHW (row-major) strides with dims {N=1, C=2, H=3, W=4}:
    // N stride = C*H*W = 2*3*4 = 24
    // C stride = H*W = 3*4 = 12
    // H stride = W = 4
    // W stride = 1 (innermost dimension)
    EXPECT_EQ(tensor.memory().count(), 24);
    EXPECT_EQ(tensor.strides()[0], 24);
    EXPECT_EQ(tensor.strides()[1], 12);
    EXPECT_EQ(tensor.strides()[2], 4);
    EXPECT_EQ(tensor.strides()[3], 1);
}

TEST(TestTensor, FillWithValues)
{
    Tensor<float> tensor({1, 2, 3, 4});

    tensor.fillWithValue(1.0f);
    auto buffer = tensor.memory().hostData();

    for(size_t i = 0; i < tensor.memory().count(); i++)
    {
        EXPECT_FLOAT_EQ(1.0f, buffer[i]);
    }
}

TEST(TestTensor, FillWithRandomValues)
{
    Tensor<float> tensor({1, 2, 3, 4});

    tensor.fillWithRandomValues(1.0f, 3.0f);
    auto buffer = tensor.memory().hostData();

    for(size_t i = 0; i < tensor.memory().count(); i++)
    {
        EXPECT_GE(buffer[i], 1.0f);
        EXPECT_LE(buffer[i], 3.0f);
    }
}

TEST(TestTensor, BasicNhwcUsage)
{
    Tensor<float> tensor({1, 2, 3, 4}, TensorLayout::NHWC);

    EXPECT_EQ(tensor.memory().count(), 24);
    // NHWC strides with dims {N=1, C=2, H=3, W=4}:
    // N stride = H*W*C = 3*4*2 = 24
    // C stride = 1 (innermost dimension)
    // H stride = W*C = 4*2 = 8
    // W stride = C = 2
    EXPECT_EQ(tensor.strides()[0], 24);
    EXPECT_EQ(tensor.strides()[1], 1);
    EXPECT_EQ(tensor.strides()[2], 8);
    EXPECT_EQ(tensor.strides()[3], 2);
}

TEST(TestTensor, GetAndSetHostValueNchw)
{
    Tensor<float> tensor({1, 2, 3, 4});
    tensor.fillWithValue(0.0f);
    tensor.setHostValue(99.0f, 0, 1, 1, 2);

    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 1, 1, 2), 99.0f);
}

TEST(TestTensor, GetAndSetHostValueNhwc)
{
    Tensor<float> tensor({1, 2, 3, 4}, TensorLayout::NHWC);
    tensor.fillWithValue(0.0f);
    tensor.setHostValue(99.0f, 0, 1, 1, 2);

    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 1, 1, 2), 99.0f);
}

TEST(TestTensor, ExceedDimensions)
{
    Tensor<float> tensor({1, 2, 3, 4});
    tensor.fillWithValue(0.0f);

    EXPECT_THROW(tensor.setHostValue(99.0f, 0, 1, 1, 2, 3), std::invalid_argument);
}
