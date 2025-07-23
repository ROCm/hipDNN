// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/test_tensor.hpp>

using namespace hipdnn_sdk::reference_test_utilities;
using namespace hipdnn_sdk::utilities;

TEST(TestTensor, BasicRowMajorUsage)
{
    Test_tensor tensor = Test_tensor::make_test_tensor<float>({1,2,3,4});

    EXPECT_EQ(tensor.memory().count(), 24);
    EXPECT_EQ(tensor.strides()[0], 24);
    EXPECT_EQ(tensor.strides()[1], 12);
    EXPECT_EQ(tensor.strides()[2], 4);
    EXPECT_EQ(tensor.strides()[3], 1);
}

TEST(TestTensor, BasicColumnMajorUsage)
{
    Test_tensor tensor = Test_tensor::make_test_tensor<float>({1,2,3,4}, false);

    EXPECT_EQ(tensor.memory().count(), 24);
    EXPECT_EQ(tensor.strides()[0], 1);
    EXPECT_EQ(tensor.strides()[1], 1);
    EXPECT_EQ(tensor.strides()[2], 2);
    EXPECT_EQ(tensor.strides()[3], 6);
}