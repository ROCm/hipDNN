// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

using namespace hipdnn_sdk::utilities;

TEST(TestTensor, BasicRowMajorUsage)
{
    SKIP_IF_NO_DEVICES();

    auto tensor = Tensor::make_tensor<float>({1, 2, 3, 4});

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

TEST(TestTensor, FillWithValuesUsage)
{
    SKIP_IF_NO_DEVICES();

    auto tensor = Tensor::make_tensor<float>({1, 2, 3, 4});

    tensor.fill_with_value(1.0f);
    auto buffer = tensor.memory().host_data<float>();

    for(size_t i = 0; i < tensor.memory().count(); i++)
    {
        EXPECT_FLOAT_EQ(1.0f, buffer[i]);
    }
}

TEST(TestTensor, FillWithRandomValuesUsage)
{
    SKIP_IF_NO_DEVICES();

    auto tensor = Tensor::make_tensor<float>({1, 2, 3, 4});

    tensor.fill_with_random_values(1.0f, 3.0f);
    auto buffer = tensor.memory().host_data<float>();

    for(size_t i = 0; i < tensor.memory().count(); i++)
    {
        EXPECT_GE(buffer[i], 1.0f);
        EXPECT_LE(buffer[i], 3.0f);
    }
}

TEST(TestTensor, BasicNHWCUsage)
{
    SKIP_IF_NO_DEVICES();

    auto tensor = Tensor::make_tensor<float>({1, 2, 3, 4}, Tensor_layout::NHWC);

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

TEST(TestTensor, GetAndSetHostValueNCHW)
{
    SKIP_IF_NO_DEVICES();

    auto tensor = Tensor::make_tensor<float>({1, 2, 3, 4});
    tensor.fill_with_value(0.0f);
    tensor.set_host_value<float>(0, 1, 1, 2, 99.0f);

    EXPECT_FLOAT_EQ(tensor.get_host_value<float>(0, 1, 1, 2), 99.0f);
}

TEST(TestTensor, GetAndSetHostValueNHWC)
{
    SKIP_IF_NO_DEVICES();

    auto tensor = Tensor::make_tensor<float>({1, 2, 3, 4}, Tensor_layout::NHWC);
    tensor.fill_with_value(0.0f);
    tensor.set_host_value<float>(0, 1, 1, 2, 99.0f);

    EXPECT_FLOAT_EQ(tensor.get_host_value<float>(0, 1, 1, 2), 99.0f);
}
