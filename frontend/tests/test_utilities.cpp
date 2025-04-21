// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/utilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(UtilitiesTests, FindCommonShapeValid)
{
    std::vector<std::vector<int64_t>> input_shapes = {{1, 2, 3}, {1, 2, 1}, {1, 1, 3}};
    std::vector<int64_t>              common_shape;

    auto error = find_common_shape(input_shapes, common_shape);
    EXPECT_EQ(error.code, error_code_t::OK);
    EXPECT_EQ(common_shape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(UtilitiesTests, FindCommonShapeEmptyInput)
{
    std::vector<std::vector<int64_t>> input_shapes = {};
    std::vector<int64_t>              common_shape;

    auto error = find_common_shape(input_shapes, common_shape);
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(UtilitiesTests, FindCommonShapeIncompatibleShapes)
{
    std::vector<std::vector<int64_t>> input_shapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}};
    std::vector<int64_t>              common_shape;

    auto error = find_common_shape(input_shapes, common_shape);
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(UtilitiesTests, FindCommonShapeSingleInput)
{
    std::vector<std::vector<int64_t>> input_shapes = {{1, 2, 3}};
    std::vector<int64_t>              common_shape;

    auto error = find_common_shape(input_shapes, common_shape);
    EXPECT_EQ(error.code, error_code_t::OK);
    EXPECT_EQ(common_shape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(UtilitiesTests, GenerateStridesNHWCValid)
{
    std::vector<int64_t> dim          = {1, 2, 3, 4};
    std::vector<int64_t> stride_order = {3, 0, 2, 1}; // NHWC
    auto                 strides      = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 1, 8, 2}));
}

TEST(UtilitiesTests, GenerateStridesNDHWCValid)
{
    std::vector<int64_t> dim          = {1, 2, 3, 4, 5};
    std::vector<int64_t> stride_order = {4, 0, 3, 2, 1}; // NDHWC
    auto                 strides      = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 1, 40, 10, 2}));
}

TEST(UtilitiesTests, GenerateStridesNCHWValid)
{
    std::vector<int64_t> dim          = {1, 2, 3, 4};
    std::vector<int64_t> stride_order = {3, 2, 1, 0}; // NCHW
    auto                 strides      = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 12, 4, 1}));
}

TEST(UtilitiesTests, GenerateStridesNCDHWValid)
{
    std::vector<int64_t> dim          = {1, 2, 3, 4, 5};
    std::vector<int64_t> stride_order = {4, 3, 2, 1, 0}; // NCDHW
    auto                 strides      = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 60, 20, 5, 1}));
}

TEST(UtilitiesTests, GenerateStridesSingleDimension)
{
    std::vector<int64_t> dim          = {5};
    std::vector<int64_t> stride_order = {0};
    auto                 strides      = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{1}));
}

TEST(UtilitiesTests, StrideOrderNHWCFiveDimensions)
{
    size_t num_dims     = 5;
    auto   stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{4, 0, 3, 2, 1}));
}

TEST(UtilitiesTests, StrideOrderNHWCFourDimensions)
{
    size_t num_dims     = 4;
    auto   stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{3, 0, 2, 1}));
}

TEST(UtilitiesTests, StrideOrderNHWCThreeDimensions)
{
    size_t num_dims     = 3;
    auto   stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{2, 0, 1}));
}

TEST(UtilitiesTests, StrideOrderNHWCWithTwoDimensions)
{
    size_t num_dims     = 2;
    auto   stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{1, 0}));
}

TEST(UtilitiesTests, StrideOrderNHWCWithSingleDimension)
{
    size_t num_dims     = 1;
    auto   stride_order = stride_order_nhwc(num_dims);

    EXPECT_EQ(stride_order, (std::vector<int64_t>{0}));
}

TEST(UtilitiesTests, GenerateStridesEmptyDimensions)
{
    std::vector<int64_t> dim          = {};
    std::vector<int64_t> stride_order = {};
    auto                 strides      = generate_strides(dim, stride_order);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}
