// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/ShapeUtils.hpp>

using namespace hipdnn_sdk::utilities;

TEST(TestShapeUtils, GenerateStridesNhwcValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> strideOrder = {3, 0, 2, 1}; // NHWC
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 1, 8, 2}));
}

TEST(TestShapeUtils, GenerateStridesNdhwcValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    std::vector<int64_t> strideOrder = {4, 0, 3, 2, 1}; // NDHWC
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 1, 40, 10, 2}));
}

TEST(TestShapeUtils, GenerateStridesNchwValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4};
    std::vector<int64_t> strideOrder = {3, 2, 1, 0}; // NCHW
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{24, 12, 4, 1}));
}

TEST(TestShapeUtils, GenerateStridesNcdhwValid)
{
    std::vector<int64_t> dim = {1, 2, 3, 4, 5};
    std::vector<int64_t> strideOrder = {4, 3, 2, 1, 0}; // NCDHW
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{120, 60, 20, 5, 1}));
}

TEST(TestShapeUtils, GenerateStridesSingleDimension)
{
    std::vector<int64_t> dim = {5};
    std::vector<int64_t> strideOrder = {0};
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{1}));
}

TEST(TestShapeUtils, StrideOrderNhwcFiveDimensions)
{
    size_t numDims = 5;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{4, 0, 3, 2, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcFourDimensions)
{
    size_t numDims = 4;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{3, 0, 2, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcThreeDimensions)
{
    size_t numDims = 3;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{2, 0, 1}));
}

TEST(TestShapeUtils, StrideOrderNhwcWithTwoDimensions)
{
    size_t numDims = 2;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{1, 0}));
}

TEST(TestShapeUtils, StrideOrderNhwcWithSingleDimension)
{
    size_t numDims = 1;
    auto strideOrder = strideOrderNhwc(numDims);

    EXPECT_EQ(strideOrder, (std::vector<int64_t>{0}));
}

TEST(TestShapeUtils, GenerateStridesEmptyDimensions)
{
    std::vector<int64_t> dim = {};
    std::vector<int64_t> strideOrder = {};
    auto strides = generateStrides(dim, strideOrder);

    EXPECT_EQ(strides, (std::vector<int64_t>{}));
}
