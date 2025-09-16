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

TEST(TestTensor, GetIndex)
{
    //Strides {60, 20, 5, 1}
    Tensor<float> tensor({2, 3, 4, 5});

    // 1*60 + 2*20 + 3*5 + 4*1 = 119
    EXPECT_EQ(tensor.getIndex(1, 2, 3, 4), 119);
    std::vector<int64_t> indices1 = {1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices1), 119);

    // Test first element
    EXPECT_EQ(tensor.getIndex(0, 0, 0, 0), 0);
    std::vector<int> indices2 = {0, 0, 0, 0};
    EXPECT_EQ(tensor.getIndex(indices2), 0);

    // Test partial indices
    EXPECT_EQ(tensor.getIndex(1, 0), 60);
    std::vector<size_t> indices3 = {1, 0};
    EXPECT_EQ(tensor.getIndex(indices3), 60);

    EXPECT_EQ(tensor.getIndex(0, 1), 20);
    std::vector<size_t> indices4 = {0, 1};
    EXPECT_EQ(tensor.getIndex(indices4), 20);
}

TEST(TestTensor, GetIndexNhwc)
{
    //Strides {60, 1, 15, 3}
    Tensor<float> tensor({2, 3, 4, 5}, TensorLayout::NHWC);

    // 1*60 + 2*1 + 3*15 + 4*3 = 119
    EXPECT_EQ(tensor.getIndex(1, 2, 3, 4), 119);
    std::vector<int64_t> indices = {1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices), 119);
}

TEST(TestTensor, GetIndexEdgeCases)
{
    Tensor<float> tensor({10, 20});

    // Last element
    EXPECT_EQ(tensor.getIndex(9, 19), 199);

    std::vector<int64_t> indices = {9, 19};
    EXPECT_EQ(tensor.getIndex(indices), 199);

    // Single dim
    EXPECT_EQ(tensor.getIndex(5), 100);
    indices = {5};
    EXPECT_EQ(tensor.getIndex(indices), 100);

    // Empty dim
    std::vector<int64_t> emptyIndices;
    EXPECT_EQ(tensor.getIndex(emptyIndices), 0);
}

TEST(TestTensor, GetIndexInvalidCases)
{
    Tensor<float> tensor({2, 3, 4});

    EXPECT_THROW(tensor.getIndex(0, 1, 2, 3), std::invalid_argument);

    std::vector<int64_t> tooManyIndices = {0, 1, 2, 3};
    EXPECT_THROW(tensor.getIndex(tooManyIndices), std::invalid_argument);
}

TEST(TestTensor, GetHostValueWithVector)
{
    Tensor<float> tensor({2, 3, 4});
    tensor.fillWithValue(0.0f);

    tensor.setHostValue(42.0f, 1, 2, 3);
    std::vector<size_t> indices2 = {1, 2, 3};
    EXPECT_FLOAT_EQ(tensor.getHostValue(indices2), 42.0f);
}

TEST(TestTensor, SetHostValueWithVector)
{
    Tensor<float> tensor({2, 3, 4});
    tensor.fillWithValue(0.0f);

    std::vector<int64_t> indices1 = {0, 1, 2};
    tensor.setHostValue(10.0f, indices1);

    std::vector<size_t> indices2 = {1, 0, 3};
    tensor.setHostValue(20.0f, indices2);

    std::vector<int> indices3 = {1, 2, 1};
    tensor.setHostValue(30.0f, indices3);

    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 1, 2), 10.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 0, 3), 20.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 2, 1), 30.0f);
}

TEST(TestTensor, MixedIndexingMethods)
{
    Tensor<float> tensor({3, 4, 5});
    tensor.fillWithValue(0.0f);

    tensor.setHostValue(100.0f, 2, 3, 4);
    std::vector<int64_t> indices = {2, 3, 4};
    EXPECT_FLOAT_EQ(tensor.getHostValue(indices), 100.0f);

    indices = {1, 2, 3};
    tensor.setHostValue(200.0f, indices);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 2, 3), 200.0f);
}

TEST(TestTensor, GetIndexConsistency)
{
    Tensor<float> tensor({5, 6, 7});

    for(int i = 0; i < 5; ++i)
    {
        for(int j = 0; j < 6; ++j)
        {
            for(int k = 0; k < 7; ++k)
            {
                auto variadicIndex = tensor.getIndex(i, j, k);

                std::vector<int64_t> vecIndices = {i, j, k};
                auto vectorIndex = tensor.getIndex(vecIndices);

                EXPECT_EQ(variadicIndex, vectorIndex);
            }
        }
    }
}
