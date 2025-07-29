// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/shape_utils.hpp>

using namespace hipdnn_sdk::utilities;

TEST(ShapeUtilitiesTests, GenerateStridesNHWCValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> stride_order = {3, 0, 2, 1}; // NHWC
    auto strides = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 1, 8, 2}));
}

TEST(ShapeUtilitiesTests, GenerateStridesNDHWCValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    std::vector<int64_t> stride_order = {4, 0, 3, 2, 1}; // NDHWC
    auto strides = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 1, 40, 10, 2}));
}

TEST(ShapeUtilitiesTests, GenerateStridesNCHWValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> stride_order = {3, 2, 1, 0}; // NCHW
    auto strides = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 12, 4, 1}));
}

TEST(ShapeUtilitiesTests, GenerateStridesNCDHWValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    std::vector<int64_t> stride_order = {4, 3, 2, 1, 0}; // NCDHW
    auto strides = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 60, 20, 5, 1}));
}

TEST(ShapeUtilitiesTests, GenerateStridesSingleDimension)
{
    std::vector<int64_t> dim = {5};
    std::vector<int64_t> stride_order = {0};
    auto strides = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{1}));
}

TEST(ShapeUtilitiesTests, StrideOrderNHWCFiveDimensions)
{
    size_t num_dims = 5;
    auto stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{4, 0, 3, 2, 1}));
}

TEST(ShapeUtilitiesTests, StrideOrderNHWCFourDimensions)
{
    size_t num_dims = 4;
    auto stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{3, 0, 2, 1}));
}

TEST(ShapeUtilitiesTests, StrideOrderNHWCThreeDimensions)
{
    size_t num_dims = 3;
    auto stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{2, 0, 1}));
}

TEST(ShapeUtilitiesTests, StrideOrderNHWCWithTwoDimensions)
{
    size_t num_dims = 2;
    auto stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{1, 0}));
}

TEST(ShapeUtilitiesTests, StrideOrderNHWCWithSingleDimension)
{
    size_t num_dims = 1;
    auto stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{0}));
}

TEST(ShapeUtilitiesTests, GenerateStridesEmptyDimensions)
{
    std::vector<int64_t> dim = {};
    std::vector<int64_t> stride_order = {};
    auto strides = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}
